import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

def detokenize(text):
    """Remove PTB style tokenization artifacts as mentioned in the paper"""
    text = re.sub(r" (\'s|\'re|\'ve|n\'t|\'ll|\'m|\'d|\')", r"\1", text)
    text = re.sub(r"(\d) : (\d)", r"\1:\2", text)
    text = re.sub(r" ([.,:;?!])", r"\1", text)
    return text

def compute_option_probability(context, before_placeholder, option, after_placeholder, model, tokenizer, device):
    """
    Compute the probability of an option and the rest of the sentence
    conditioned on the context, following the approach described in the paper.
    """
    # Create the complete text with this option
    completed_question = before_placeholder + option + after_placeholder
    
    # Detokenize
    context_detok = detokenize(context)
    completed_question_detok = detokenize(completed_question)
    
    # Full text: context + completed question
    full_text = context_detok + " " + completed_question_detok
    
    # Tokenize
    def _encode(text):
        return torch.tensor(tokenizer.encode(text)).to(device).view(1, -1)
    inputs = _encode(full_text)
    
    # Find where the option and the rest of the sentence start in the tokenized text
    option_position_text = context_detok + " " + detokenize(before_placeholder)
    option_position_ids = _encode(option_position_text)
    option_start = option_position_ids.shape[1] - 1  # -1 to account for the shift in log probs
    
    # Get the logits from the model
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs
    
    # Shift logits and input IDs for computing probabilities (predicting next token)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ids = inputs[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Get log probabilities of the actual tokens
    token_log_probs = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum the log probabilities of the option and everything after it
    option_and_rest_log_prob = token_log_probs[0, option_start:].sum().item()
    
    return option_and_rest_log_prob

def evaluate_downstream_cbt_with_probs(model, tokenizer, device, dataset_split="validation", verbose=True, max_examples=None, max_context_length=1024):
    """
    Evaluate a model on the CBT dataset with detailed probability outputs.
    
    Args:
        model: The model to evaluate (must be a GPT-2 compatible model)
        tokenizer: The tokenizer to use with the model
        device: The device (cuda/cpu) to run on
        dataset_split: The dataset split to evaluate on ("validation" or "test")
        verbose: Whether to print detailed outputs
        max_examples: Optional limit on number of examples to process
        max_context_length: Maximum allowed context length in tokens (examples exceeding this will be skipped)
    
    Returns:
        Accuracy of the model on the given dataset split
    """
    print(f"Evaluating model on CBT-CN {dataset_split} set...")
    print(f"Max context length: {max_context_length} tokens")
    
    # Load dataset
    dataset = load_dataset("cbt", "CN", split=dataset_split)
    
    correct = 0
    total = 0
    skipped_examples = 0
    
    # Limit dataset size if specified
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    all_results = []
    
    for example in tqdm(dataset, desc=f"Processing {dataset_split} examples"):
        # Get the context, question, options, and correct answer
        context = " ".join(example['sentences'])
        question = example['question']
        options = example['options']
        correct_answer = example['answer']
        
        # Skip examples without the placeholder
        if 'XXXXX' not in question:
            continue
        
        # Get the parts before and after the placeholder
        parts = question.split('XXXXX')
        if len(parts) != 2:
            continue
        
        before_placeholder, after_placeholder = parts
        
        # Check context length with the longest option to avoid tokenizing multiple times
        longest_option = max(options, key=len)
        full_sample = context + " " + before_placeholder + longest_option + after_placeholder
        
        # Get token count
        token_count = len(tokenizer.encode(full_sample))
        
        # Skip if context is too long
        if token_count > max_context_length:
            skipped_examples += 1
            if verbose:
                print(f"\nSkipping example (token count: {token_count} > {max_context_length}):")
                print(f"Question: {question}")
                print("-" * 80)
            continue
        
        # Compute probabilities for each option
        option_scores = []
        for option in options:
            score = compute_option_probability(
                context, before_placeholder, option, after_placeholder, 
                model, tokenizer, device
            )
            option_scores.append(score)
        
        # Convert log probabilities to normalized probabilities for easier interpretation
        log_probs = np.array(option_scores)
        max_log_prob = np.max(log_probs)
        
        # Subtract max for numerical stability before exp
        exp_log_probs = np.exp(log_probs - max_log_prob)
        probs = exp_log_probs / np.sum(exp_log_probs)
        
        # Sort options by probability
        sorted_indices = np.argsort(-probs)  # Sort in descending order
        sorted_options = [options[i] for i in sorted_indices]
        sorted_probs = [probs[i] for i in sorted_indices]
        is_correct = [options[i] == correct_answer for i in sorted_indices]
        
        # Choose the option with the highest probability
        best_option_idx = np.argmax(option_scores)
        predicted_answer = options[best_option_idx]
        
        # Check if correct
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
        
        # Save results for this example
        result = {
            'id': total,
            'token_count': token_count,
            'context': context[-100:] + "...",  # Truncated context for display
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': predicted_answer == correct_answer,
            'options_by_prob': list(zip(sorted_options, sorted_probs, is_correct))
        }
        all_results.append(result)
        
        # Print detailed results if verbose
        if verbose:
            print(f"\nExample {total} (tokens: {token_count}):")
            print(f"Question: {question}")
            print(f"Correct answer: {correct_answer}")
            print(f"Predicted answer: {predicted_answer}")
            print(f"Correct? {'✓' if predicted_answer == correct_answer else '✗'}")
            
            # Display all options with their probabilities
            rows = []
            for opt, prob, is_opt_correct in result['options_by_prob']:
                rows.append([
                    opt, 
                    f"{prob:.2%}", 
                    "✓" if is_opt_correct else ""
                ])
            
            print(tabulate(rows, headers=["Option", "Probability", "Correct"], tablefmt="simple"))
            # Fix for current accuracy calculation
            accuracy_pct = (correct/total)*100 if total > 0 else 0
            print(f"Current accuracy: {accuracy_pct:.2f}% ({correct}/{total})")
            print("-" * 80)
    
    final_accuracy = correct / total if total > 0 else 0
    print(f"\nFinal accuracy on {dataset_split}: {final_accuracy:.2%} ({correct}/{total})")
    print(f"Skipped examples due to context length: {skipped_examples}")
    
    # Show summary of top-k accuracy
    top_k_correct = [0] * 5
    
    for result in all_results:
        for k in range(min(5, len(result['options_by_prob']))):
            if any(correct for _, _, correct in result['options_by_prob'][:k+1]):
                top_k_correct[k] += 1
    
    print("\nTop-K Accuracy:")
    for k in range(5):
        if k < len(top_k_correct):
            print(f"Top-{k+1} accuracy: {top_k_correct[k]/total:.2%}")
    
    # Show token count statistics
    if all_results:
        token_counts = [result['token_count'] for result in all_results]
        print("\nToken Count Statistics:")
        print(f"Min: {min(token_counts)}")
        print(f"Max: {max(token_counts)}")
        print(f"Mean: {sum(token_counts)/len(token_counts):.1f}")
        print(f"Median: {sorted(token_counts)[len(token_counts)//2]}")
    
    return final_accuracy, all_results, skipped_examples


# EXAMPLE 1: Using a pre-trained model from HuggingFace
def load_pretrained_model(model_name="gpt2"):
    """
    Load a pre-trained model from HuggingFace
    """
    print(f"Loading pre-trained model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


# Example usage with pre-trained model
def run_pretrained_evaluation():
    """Example of evaluating a pre-trained model from HuggingFace"""
    model_name = "gpt2-medium"  # Can be "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
    
    # Load the model
    model, tokenizer, device = load_pretrained_model(model_name)
    
    # Get model's max context length
    max_length = model.config.n_ctx if hasattr(model.config, 'n_ctx') else 1024
    print(f"Model's maximum context length: {max_length}")
    
    # Run evaluation with context length limit
    accuracy, results, skipped = evaluate_cbt_with_probs(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset_split="validation",
        verbose=True,
        max_examples=50,  # Set to None to evaluate all examples
        max_context_length=max_length  # Use model's max context length
    )
    
    print(f"\nEvaluation complete for {model_name}!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Skipped examples: {skipped}")


if __name__ == "__main__":
    # Run the evaluation
    run_pretrained_evaluation()