class Node:

    def __init__(self, letter, parent):
        self.sorted_sentences = []
        self.letter = letter
        self.children = {}
        self.parent = parent

    def node_print(self):
        print(self.letter)
        print(self.children)
        print(self.sorted_sentences)
        print("\n")
        for v in self.children.values():
            v.node_print()
    
    def next(self, letter):
        if letter not in self.children.keys():
            self.children[letter] = Node(letter, self)
        return self.children[letter]

    def backtrack(self, new_sentence, input_hotness=1):
        new_hotness = input_hotness

        if not self.sorted_sentences:
            self.sorted_sentences = [(new_sentence, new_hotness)]
        
        else:
            for idx, (sentence, hotness) in enumerate(self.sorted_sentences):
                if sentence == new_sentence:
                    new_hotness += hotness
                    self.sorted_sentences.pop(idx)

            self.sorted_sentences.append(("LAST", -1))
            for idx, (sentence, hotness) in enumerate(self.sorted_sentences):
                if hotness < new_hotness:
                    self.sorted_sentences.insert(idx, (new_sentence, new_hotness))
            self.sorted_sentences = self.sorted_sentences[:-1] 
        
        if self.letter != "ROOT":
            self.parent.backtrack(new_sentence)
        

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
            node_iterator.backtrack(sentence, times[idx])
        
        #self.root.node_print()
        
        
    def input(self, c: str) -> List[str]:

        if c == "#":
            self.current_node.backtrack(self.sentence)
            self.sentence = ""
            self.current_node = self.root
            return []

        self.current_node = self.current_node.next(c)
        self.sentence += c
        top_sorted_sentences = self.current_node.sorted_sentences[:3]
        return [sentence for sentence, _ in top_sorted_sentences]

# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)
