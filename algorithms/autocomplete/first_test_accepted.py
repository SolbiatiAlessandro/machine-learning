from dataclasses import dataclass

@dataclass
class SentencePair:
    sentence: str
    hotness: float

SentencePair.__lt__ = lambda self, other: self.hotness > other.hotness if self.hotness != other.hotness else self.sentence < other.sentence

class Node:
    def __init__(self, letter, parent):
        self.sorted_sentences = []
        self.letter = letter
        self.children = {}
        self.parent = parent

    def node_print(self):
        print(self.letter, self.children, self.sorted_sentences)
        for v in self.children.values():
            v.node_print()
    
    def next(self, letter):
        if letter not in self.children.keys():
            self.children[letter] = Node(letter, self)
        return self.children[letter]

    def backtrack(self, new_sentence, input_hotness=1):
        new_hotness = input_hotness

        for idx, sentence_pair in enumerate(self.sorted_sentences):
            if sentence_pair.sentence == new_sentence:
                new_hotness += sentence_pair.hotness
                self.sorted_sentences.pop(idx)

        self.sorted_sentences.append(SentencePair(new_sentence, new_hotness))
        self.sorted_sentences.sort()
    
        if self.letter != "ROOT":
            self.parent.backtrack(new_sentence, new_hotness)

class AutocompleteSystem:
    # prefix tree 

    def __init__(self, sentences: List[str], times: List[int]):

        self.root = Node("ROOT", None)
        self.current_node = self.root
        self.sentence = ""
        
        for idx, sentence in enumerate(sentences):
            node_iterator = self.current_node
            sentence = sentences[idx]
            for letter in sentence:
                node_iterator = node_iterator.next(letter)
            node_iterator.backtrack(sentence, input_hotness=times[idx])
        
        
    def input(self, c: str) -> List[str]:

        if c == "#":
            self.current_node.backtrack(self.sentence)
            self.sentence = ""
            self.current_node = self.root
            return []

        self.current_node = self.current_node.next(c)
        self.sentence += c
        top_sorted_sentences = self.current_node.sorted_sentences[:3]

        return [sentence_pair.sentence for sentence_pair in top_sorted_sentences]

# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)
