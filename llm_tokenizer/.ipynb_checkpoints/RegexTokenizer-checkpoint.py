"""Regex + BPE Tokenizer to implement GPT2 and GPT4 tokenizers from Karpathy tokenizers class"""
from dataclasses import dataclass
import regex as re


@dataclass
class TokenPair:
    first_token: int
    second_token: int
    frequency: int = 1
    
    def _key(self):
        return (self.first_token, self.second_token)
    
    def __lt__(self, other):
        return self.frequency < other.frequency
    
    def to_string(self):
        return chr(self.first_token) + chr(self.second_token)


class Tokenizer:
    """Regex + BPE Tokenizer to implement GPT2 and GPT4 tokenizers from Karpathy tokenizers class
    
    
    tokenizer = Tokenizer("the pen is on the table")
    """
    def __init__(
        self, 
        text, 
        encoding_vocab_size=276,  
        name="tinyshakespeare", 
        path_prefix=None,
        regex=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ):
        """
        tokens: str
        """

        self.name = name
        self.path_prefix=path_prefix
        self.encoding_vocab_size = encoding_vocab_size
        self._regex = regex
        
        self.tokens = text.encode('utf-8')
        regex_text = set(re.findall(re.compile(self._regex), text))
        print(f"[RegexTokenizer.__init__] {len(regex_text)} regex groups")
        self.regex_tokens = list(map(lambda x: x.encode('utf-8'), regex_text))
        print(f"[RegexTokenizer.__init__] regex groups encoded to utf-8")
        
        #self._original_tokens, self.encoded_tokens = tokens, tokens 
        self.mint_token = 256
        self.count()
        self.decoding_map, self.encoding_map = {}, {}
        
    def count(self):
        self.tcounts = {}
        for tokens in self.regex_tokens:
            for i, token in enumerate(tokens[:-1]):
                tp = TokenPair(token, tokens[i+1])
                if self.tcounts.get(tp._key(), None):
                    self.tcounts[tp._key()].frequency += 1
                else:
                    self.tcounts[tp._key()] = tp
        
    def get_most_common(self):
        return max(self.tcounts.values())
    
    def swap_top(self, debug=False):
        """ returns True if finished encoding """
        top_tp = self.get_most_common()
        if debug: print(top_tp)
        if top_tp.frequency == 1: return True
        a, b = top_tp.first_token, top_tp.second_token

        new_regex_tokens = []
        for tokens in self.regex_tokens:
            new_encoding, idx = [], 0
            while idx < len(tokens) - 1:
                A, B = tokens[idx], tokens[idx+1]
                if a == A and b == B:
                    new_encoding.append(self.mint_token)
                    idx += 2
                else:
                    new_encoding.append(A)
                    idx += 1
                    
            if idx < len(tokens):
                new_encoding.append(tokens[idx])
            
            new_regex_tokens.append(new_encoding)
            
        
        self.regex_tokens = new_regex_tokens
        self.decoding_map[self.mint_token] = top_tp
        self.encoding_map[top_tp._key()] = self.mint_token
        
        
        if debug and self.mint_token % 256 == 0 : print(f"[Tokenizer.swap_top] {self.mint_token}")
        self.mint_token += 1
        self.count()
        if debug: print(self.regex_tokens)
        return self.mint_token == self.encoding_vocab_size
            
    def train(self, debug=False):
        """returns the encoded training set"""
        finshed_encoding = self.swap_top(debug=debug)
        while not finshed_encoding:
            finshed_encoding = self.swap_top(debug=debug)
            if self.mint_token % 100 == 0: print(f"[BPETokenizer.train] mint_token={self.mint_token}")
        #if debug: print(self.encoded_tokens)
        return self.regex_tokens
    
    def _filename(self):
        res = f"tokenizer_{self.name}.pickle"
        if self.path_prefix:
            res = self.path_prefix + res
        return res

    def save_to_file(self):     
        import pickle

        # An arbitrary collection of objects supported by pickle.
        data = {
            'encoding_map': self.encoding_map,
            'decoding_map': self.decoding_map,
        }

        with open(self._filename(), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            
    def load_from_file(self):
        import pickle

        with open(self._filename(), 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            self.encoding_map = data['encoding_map']
            self.decoding_map = data['decoding_map']
        
    
    def decode(self, encoded_tokens, debug=False, raw_tokens=True):
        decoded = False
        
        while not decoded:
            decoded = True
            decoded_tokens = []
            for token in encoded_tokens:
                tp = self.decoding_map.get(token, None)
                if tp:
                    decoded = False
                    decoded_tokens.append(tp.first_token)
                    decoded_tokens.append(tp.second_token)
                else:
                    decoded_tokens.append(token)
            encoded_tokens = decoded_tokens
        if debug: print(decoded_tokens)
        if raw_tokens:
            return decoded_tokens
        return bytes(decoded_tokens).decode('utf-8', errors="replace")
    
    def encode(self, text, debug=False):
        decoded_tokens = list(text.encode("utf-8"))
        encoded = False
        
        while not encoded:
            encoded, idx = True, 0
            encoded_tokens = []
            while idx < len(decoded_tokens) - 1:
                tp = TokenPair(decoded_tokens[idx], decoded_tokens[idx+1])
                encoded_token = self.encoding_map.get(tp._key(), None)
                if encoded_token:
                    encoded_tokens.append(encoded_token)
                    idx += 2
                    encoded = False
                else:
                    encoded_tokens.append(tp.first_token)
                    idx += 1
            if idx < len(decoded_tokens): encoded_tokens.append(decoded_tokens[idx])
            decoded_tokens = encoded_tokens
            
        if debug: print(encoded_tokens)
        return encoded_tokens
    
    def visualize_encoding_map(self):
        """Visualizes the merges in the encoding map as 'ab' 'cd' -> 'abcd'."""
        print("Encoding Map Visualization:")
        for minted_token, token_pair in sorted(self.decoding_map.items()):
            def decode_token(token):
                if token < 256:
                    return chr(token) if 32 <= token < 127 else f"[{token}]"
                elif token in self.decoding_map:
                    first = decode_token(self.decoding_map[token].first_token)
                    second = decode_token(self.decoding_map[token].second_token)
                    return first + second
                else:
                    return f"[{token}]"

            first = decode_token(token_pair.first_token)
            second = decode_token(token_pair.second_token)
            merged = first + second
            print(f"'{first}' '{second}' -> '{merged}' (Token {minted_token})")
            
            
            
from dataclasses import dataclass
import regex as re
import heapq
import pickle

@dataclass
class TokenPair:
    first_token: int
    second_token: int
    frequency: int = 1  # not used in the incremental version

    def _key(self):
        return (self.first_token, self.second_token)

    def __lt__(self, other):
        return self.frequency < other.frequency

    def to_string(self):
        return chr(self.first_token) + chr(self.second_token)

class FastTokenizer:
    """
    Fast tokenizer optimized for speed, not for readibility
    
    
    Regex+BPE Tokenizer (GPT2/GPT4 style) with incremental frequency updates
    and a heap for fast maximum lookup.
    
    This version converts tokens to lists of ints for consistency and
    updates only the sequences that change after each merge.
    """
    def __init__(
        self, 
        text, 
        encoding_vocab_size=276,  
        name="tinyshakespeare", 
        path_prefix=None,
        regex=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    ):
        self.name = name
        self.path_prefix = path_prefix
        self.encoding_vocab_size = encoding_vocab_size
        self._regex = regex

        # tokenize the text using the regex; note that we use all matches (order preserved)
        tokens = re.findall(self._regex, text)
        # for demonstration, we’ll use each regex match in order; 
        # convert each match to its UTF-8 encoding and then to a list of ints.
        self.regex_tokens = [list(x.encode('utf-8')) for x in tokens]
        print(f"[Tokenizer.__init__] {len(self.regex_tokens)} regex tokens (order preserved)")
        
        # Start minting new tokens after 255 (assuming 0-255 are reserved)
        self.mint_token = 256
        self.decoding_map = {}
        self.encoding_map = {}

        # For incremental pair frequency updates, we maintain:
        # 1. For each sequence (regex token) its adjacent pairs (with positions)
        # 2. A global mapping from a pair (tuple) to a set of (sequence_index, position)
        # 3. A heap for quick lookup of the most frequent pair.
        self.sequence_pair_occurrences = {}  # seq_index -> list of (position, (a,b))
        self.pair_occurrences = {}  # (a,b) -> set of (seq_index, pos)
        self.heap = []  # entries are (-frequency, (a,b))
        self._build_all_occurrences()

    def _build_all_occurrences(self):
        """Initializes pair occurrences for every sequence."""
        for seq_index in range(len(self.regex_tokens)):
            self._rebuild_occurrences(seq_index)

    def _rebuild_occurrences(self, seq_index):
        """
        For a given sequence index, remove any old pair occurrences and rebuild them.
        Update the global pair_occurrences dict and push new counts onto the heap.
        """
        # Remove old occurrences for this sequence
        if seq_index in self.sequence_pair_occurrences:
            for pos, pair in self.sequence_pair_occurrences[seq_index]:
                if pair in self.pair_occurrences:
                    self.pair_occurrences[pair].discard((seq_index, pos))
                    if not self.pair_occurrences[pair]:
                        del self.pair_occurrences[pair]
            self.sequence_pair_occurrences[seq_index] = []

        seq = self.regex_tokens[seq_index]
        occ_list = []
        for pos in range(len(seq) - 1):
            pair = (seq[pos], seq[pos+1])
            occ_list.append((pos, pair))
            if pair not in self.pair_occurrences:
                self.pair_occurrences[pair] = set()
            self.pair_occurrences[pair].add((seq_index, pos))
            # push new count onto the heap
            heapq.heappush(self.heap, (-len(self.pair_occurrences[pair]), pair))
        self.sequence_pair_occurrences[seq_index] = occ_list

    def swap_top(self, debug=False):
        """Perform one merge of the most frequent pair using incremental updates.
           Returns True if finished encoding (no pair frequency > 1 or target vocab reached).
        """
        # Pop from heap until we find a pair whose frequency is up to date.
        while self.heap:
            neg_freq, pair = heapq.heappop(self.heap)
            freq = -neg_freq
            # Check if this pair still exists with the same frequency.
            if pair not in self.pair_occurrences or len(self.pair_occurrences[pair]) != freq:
                continue  # stale entry; skip
            if freq == 1:
                # No pair appears more than once.
                return True
            a, b = pair
            if debug:
                print(f"[swap_top] Merging pair {pair} occurring {freq} times")
            new_token = self.mint_token
            self.mint_token += 1
            self.decoding_map[new_token] = TokenPair(a, b)
            self.encoding_map[pair] = new_token

            # Get all affected sequences
            affected = {seq_index for (seq_index, pos) in self.pair_occurrences[pair]}
            # For each affected sequence, perform a merge pass.
            for seq_index in affected:
                old_seq = self.regex_tokens[seq_index]
                new_seq = []
                i = 0
                while i < len(old_seq) - 1:
                    if old_seq[i] == a and old_seq[i+1] == b:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1
                if i < len(old_seq):
                    new_seq.append(old_seq[i])
                self.regex_tokens[seq_index] = new_seq
                # Rebuild occurrences for this sequence only.
                self._rebuild_occurrences(seq_index)

            # Remove the merged pair from the global dictionary.
            if pair in self.pair_occurrences:
                del self.pair_occurrences[pair]
            # If we’ve reached our target vocab size, stop.
            if self.mint_token == self.encoding_vocab_size:
                return True
            return False  # one merge performed

        return True  # if heap is empty, we are done

    def train(self, debug=False):
        """Train the BPE merges until no pair occurs more than once
           or until encoding_vocab_size is reached.
           Returns the final encoded sequences.
        """
        finished = self.swap_top(debug=debug)
        while not finished:
            finished = self.swap_top(debug=debug)
            if self.mint_token % 100 == 0:
                print(f"[train] mint_token = {self.mint_token}")
        return self.regex_tokens

    def save_to_file(self):     
        data = {
            'encoding_map': self.encoding_map,
            'decoding_map': self.decoding_map,
        }
        with open(self._filename(), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_from_file(self):
        with open(self._filename(), 'rb') as f:
            data = pickle.load(f)
            self.encoding_map = data['encoding_map']
            self.decoding_map = data['decoding_map']

    def _filename(self):
        res = f"tokenizer_{self.name}.pickle"
        if self.path_prefix:
            res = self.path_prefix + res
        return res

    def decode(self, encoded_tokens, debug=False, raw_tokens=True):
        decoded = False
        while not decoded:
            decoded = True
            decoded_tokens = []
            for token in encoded_tokens:
                tp = self.decoding_map.get(token, None)
                if tp:
                    decoded = False
                    decoded_tokens.append(tp.first_token)
                    decoded_tokens.append(tp.second_token)
                else:
                    decoded_tokens.append(token)
            encoded_tokens = decoded_tokens
        if debug:
            print(decoded_tokens)
        if raw_tokens:
            return decoded_tokens
        return bytes(decoded_tokens).decode('utf-8', errors="replace")

    def encode(self, text, debug=False):
        # Convert text to list of ints
        decoded_tokens = list(text.encode("utf-8"))
        encoded = False
        while not encoded:
            encoded = True
            idx = 0
            encoded_tokens = []
            while idx < len(decoded_tokens) - 1:
                pair = (decoded_tokens[idx], decoded_tokens[idx+1])
                encoded_token = self.encoding_map.get(pair, None)
                if encoded_token:
                    encoded_tokens.append(encoded_token)
                    idx += 2
                    encoded = False
                else:
                    encoded_tokens.append(decoded_tokens[idx])
                    idx += 1
            if idx < len(decoded_tokens):
                encoded_tokens.append(decoded_tokens[idx])
            decoded_tokens = encoded_tokens
            if debug:
                print(encoded_tokens)
        return encoded_tokens

    def visualize_encoding_map(self):
        """Visualizes the merges in the encoding map as 'ab' 'cd' -> 'abcd'."""
        print("Encoding Map Visualization:")
        for minted_token, token_pair in sorted(self.decoding_map.items()):
            def decode_token(token):
                if token < 256:
                    return chr(token) if 32 <= token < 127 else f"[{token}]"
                elif token in self.decoding_map:
                    first = decode_token(self.decoding_map[token].first_token)
                    second = decode_token(self.decoding_map[token].second_token)
                    return first + second
                else:
                    return f"[{token}]"
            first = decode_token(token_pair.first_token)
            second = decode_token(token_pair.second_token)
            merged = first + second
            print(f"'{first}' '{second}' -> '{merged}' (Token {minted_token})")


          
