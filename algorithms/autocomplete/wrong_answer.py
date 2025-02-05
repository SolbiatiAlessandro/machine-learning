LAST_LETTER = '#'

class Node:
    def __init__(self, letter: str, parent: Node):
        self.type = "letter"
        self.letter = letter
        self.children = []
        self.parent = parent
        self.max_hotness = None
        self.children_by_hotness = []
    
    def set_next(self, letter, hotness):
        for i, c in enumerate(self.children):
            if c.letter = letter:
                if hotness > c.hotness:
                    self.children.pop(i)
                    #insert sorted children
                else

        if letter not in self.children.keys():
            self.children[letter] = Node(letter)
        return self.children[letter]
    
    def get_next(self):
        # get hottest children
        res, highest_hotness = None, -1
        for c in self.children:
            if c.max_hotness > highest_hotness:
                res = c
        return res

    def last(self, hotness):
        if letter not in self.children.keys():
            self.children[LAST_LETTER] = EndNode(hotness)
        return self.children[LAST_LETTER]


class EndNode:
    def __init__(self, parent: Node, hotness: int):
        self.type = "end"
        self.parent = parent
        self.max_hotness = hotness
        self.sentence = ""

        #backtrack
        current_node = parent
        while parent.letter != "#":

            # optimization here keeping it sorted
            letter, highest_hotness = "", -1
            for c in current_node.children:
                if c.max_hotness > highest_hotness:
                    letter = c.letter
                    current_node = c 
            current_node = current_node.parent
            self.sentence += letter


class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        self.sentences = sentences
        self.times = times
        self.root = Node('#')
        self.current_node = self.root

        # initialize tree
        for i in len(sentences):
            current_node = self.root
            sentence = sentences[i]
            time = times[i]
            for c in sentence:
                current_node = current_node.set_next(c)
            current_node.last(time)

    def input(self, c: str) -> List[str]:
        if c == "#":
            self.current_node = self.root
            # TODO: update hotness
            return []

        self.current_node = self.current_node.set_next(c)

        autocomplete_node = self.current_node
        while autocomplete_node.type == 'letter':
            autocomplete_node = autocomplete_node.get_next()
        return autocomplete_node.sentence


        
        
        


# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)
