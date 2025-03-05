import os
import tiktoken
import pickle
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import multiprocessing
import time
from functools import partial
import torch

# Configuration
NUM_CORES = 24  # Reduced from 90 - see explanation below
SHARD_SIZE = 250000  # Number of examples per shard, adjusted for better performance
OUTPUT_DIR = "tokenized_shards"
CONTEXT_LENGTH = 1024
PROCESS_BATCH_SIZE = 5000  # Larger batch size for each process task

print(f"Setting up data processing with {NUM_CORES} parallel processes")
print(f"Using {SHARD_SIZE} examples per shard")

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to tokenize a LARGER batch of texts - more efficient for multiprocessing
def process_large_batch(examples, start_idx, end_idx):
    """Process a larger chunk of examples at once to reduce overhead"""
    # Extract the slice we're working on
    batch_texts = examples[start_idx:end_idx]["text"]
    
    # Process in one go
    tokenized = [enc.encode(text, allowed_special={"<|endoftext|>"}) for text in batch_texts]
    
    return tokenized

# Optimized function for dataset processing
def process_dataset_optimized(dataset_split, output_prefix="shard"):
    total_examples = len(dataset_split)
    print(f"Processing {total_examples:,} examples with optimized parallel approach")
    
    # Calculate number of shards
    num_shards = (total_examples + SHARD_SIZE - 1) // SHARD_SIZE
    print(f"Will create approximately {num_shards} shards")
    
    # Start timer
    start_time = time.time()
    
    # Process dataset in shards directly (one shard at a time)
    for shard_idx in range(num_shards):
        shard_start_time = time.time()
        
        # Calculate indices for this shard
        start_idx = shard_idx * SHARD_SIZE
        end_idx = min((shard_idx + 1) * SHARD_SIZE, total_examples)
        shard_size = end_idx - start_idx
        
        print(f"\nProcessing shard {shard_idx+1}/{num_shards} ({shard_size:,} examples)")
        
        # Split this shard into chunks for parallel processing
        chunks = []
        for chunk_start in range(start_idx, end_idx, PROCESS_BATCH_SIZE):
            chunk_end = min(chunk_start + PROCESS_BATCH_SIZE, end_idx)
            chunks.append((chunk_start, chunk_end))
        
        # Create a pool with the optimal number of workers
        with multiprocessing.Pool(NUM_CORES) as pool:
            # Process chunks in parallel - use partial to freeze dataset_split parameter
            process_func = partial(process_large_batch, dataset_split)
            
            # Process all chunks and track progress
            tokenized_chunks = []
            for i, result in enumerate(tqdm(
                pool.starmap(process_func, chunks), 
                total=len(chunks),
                desc="Processing chunks"
            )):
                tokenized_chunks.append(result)
        
        # Flatten all chunks for this shard
        all_tokenized = [item for sublist in tokenized_chunks for item in sublist]
        
        # Save the shard using pickle (handles variable length sequences)
        shard_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{shard_idx:05d}.pkl")
        with open(shard_path, 'wb') as f:
            pickle.dump(all_tokenized, f)
        
        shard_time = time.time() - shard_start_time
        print(f"Saved shard {shard_idx} with {len(all_tokenized):,} examples in {shard_time:.2f} seconds")
        
        # Free memory
        del all_tokenized
        del tokenized_chunks
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return num_shards

def setup():
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")
    
    # Process and shard the dataset
    print("Processing training split...")
    num_shards = process_dataset_optimized(ds["train"], "train_shard")
    print(f"Created {num_shards} shards")
    
    # Show how to load and use shards
    print("\nExample of loading a shard:")
    with open(os.path.join(OUTPUT_DIR, "train_shard_00000.pkl"), "rb") as f:
        sample_shard = pickle.load(f)
    
    print(f"Loaded shard with {len(sample_shard)} examples")
    print(f"First example length: {len(sample_shard[0])} tokens")

# Optimized dataset class for training
class PickleShardDataset(torch.utils.data.Dataset):
    def __init__(self, shard_dir, prefix="train_shard", context_length=1024, preload_shards=2):
        self.shard_dir = shard_dir
        self.context_length = context_length
        self.preload_shards = preload_shards
        
        # Get all shard files
        self.shard_files = sorted([
            f for f in os.listdir(shard_dir) if f.startswith(prefix) and f.endswith('.pkl')
        ])
        
        if not self.shard_files:
            raise ValueError(f"No shards found with prefix {prefix} in {shard_dir}")
        
        print(f"Found {len(self.shard_files)} shards")
        
        # Preload initial shards
        self.loaded_shards = {}
        self.current_shard_idx = 0
        
        # Load first shard to get metadata
        first_shard_path = os.path.join(shard_dir, self.shard_files[0])
        with open(first_shard_path, 'rb') as f:
            self.loaded_shards[0] = pickle.load(f)
        
        # Calculate total examples
        self.examples_per_shard = len(self.loaded_shards[0])
        self.total_examples = self.examples_per_shard * len(self.shard_files)
        
        print(f"Dataset contains approximately {self.total_examples:,} examples")
        print(f"Each shard contains ~{self.examples_per_shard:,} examples")
    
    def _load_shard(self, shard_idx):
        """Load a shard into memory if not already loaded"""
        if shard_idx in self.loaded_shards:
            return
        
        # Load the requested shard
        shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
        with open(shard_path, 'rb') as f:
            self.loaded_shards[shard_idx] = pickle.load(f)
        
        # If we have too many shards loaded, remove the oldest one
        if len(self.loaded_shards) > self.preload_shards:
            # Find shards to keep (current and next)
            keep_shards = {
                self.current_shard_idx,
                (self.current_shard_idx + 1) % len(self.shard_files)
            }
            
            # Remove shards we don't need
            for key in list(self.loaded_shards.keys()):
                if key not in keep_shards:
                    del self.loaded_shards[key]
                    break
    
    def __len__(self):
        return self.total_examples
    
    def __getitem__(self, idx):
        # Convert global index to shard index and local index
        shard_idx = idx // self.examples_per_shard
        local_idx = idx % self.examples_per_shard
        
        # Ensure we're using the actual available shards
        shard_idx = shard_idx % len(self.shard_files)
        
        # Load shard if needed
        if shard_idx not in self.loaded_shards:
            self._load_shard(shard_idx)
        
        # Update current shard tracking
        self.current_shard_idx = shard_idx
        
        # Get tokens from the appropriate shard
        tokens = self.loaded_shards[shard_idx][local_idx]
        
        # Process to fixed context length
        if len(tokens) > self.context_length:
            # Take a random slice for diversity
            start_idx = np.random.randint(0, len(tokens) - self.context_length)
            tokens = tokens[start_idx:start_idx + self.context_length]
        else:
            # Pad shorter sequences
            tokens = tokens + [0] * (self.context_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }

def get_dataloader(
    shard_dir=OUTPUT_DIR,
    batch_size=16,
    num_workers=12,  # Reduced worker count for more stability
    pin_memory=True
):
    """Create an optimized dataloader for training"""
    dataset = PickleShardDataset(shard_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    return dataloader

if __name__ == "__main__":
    setup()
    print("\nTesting dataloader...")
    dataloader = get_dataloader(batch_size=8)
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        break