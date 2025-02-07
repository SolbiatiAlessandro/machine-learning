class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent_from_right = None
        self.parent_from_left = None

class Queue:
    def __init__(self):
        self._queue = []
        self._debug = False

    def _print_queue(self, text):
        print(text, [n.value for n in self._queue])

    def push(self, x):
        self._queue.append(x)
        if self._debug:
            self._print_queue("push")

    def peak(self):
        return self._queue[0] if self._queue else None

    def pop(self):
        res = self._queue[0]
        self._queue = self._queue[1:]
        if self._debug:
            print(res.value)
            self._print_queue("pop")
        return res

class MinHeap:

    def __init__(self, values):
        self.root = None
        self.queue = Queue()

        for value in values:
            self.push(value)

    def _get_next_node(self) -> Node:
        next_node = self.queue.peak()
        if next_node.left and next_node.right:
            self.queue.pop()
        return next_node

    def push(self, x: int) -> None:
        if not self.root:
            self.root = Node(x)
            self.queue.push(self.root)
            return

        current = Node(x)
        parent = self.queue.peak()

        # look at the queue if parents are full pop it
        if not parent.left:
            parent.left = current
            current.parent_from_left = parent
        elif not parent.right:
            parent.right = current
            current.parent_from_right = parent
            self.queue.pop()
        self.queue.push(current)

        # swap parent and current until heap ordering is not correct
        while parent.value > current.value:
            temp = current.value
            current.value = parent.value
            parent.value = temp
            current = parent

    def pop(self) -> int:
        if not self.root: return None
        if len(self.queue) == 1:
            res = self.root.value
            self.queue.pop()
            self.root = None
            return res

        res = self.root.value
        last_node = self.queue.pop()
        if last_node.right:
            swap_value = last_node.right.value
            last_node.right = None
        elif last_node.left:
            swap_value = last_node.left.value
            last_node.left = None
        else:
            swap_value = last_node.value
            if last_node.parent_from_right:
                last_node.parent_from_right.right = None
            elif last_node.parent_from_left:
                last_node.parent_from_left.left = None
            else: # not supposed to come here cause we removed root at the top
                raise AssertionError("non-root node without parents")

        current = swap_value
        def next_child(current):
            if not current.left:
                child = current.right
            elif not current.right:
                child = current.left
            elif current.left < current.right:
                child = current.left
            else:
                child = current.right
            return child

        while current > next_child(current):
            child = next_child(current)
            temp =  current.value
            current.value =  child.value
            child.value = temp
            child = current

        return res

    def peak(self) -> int:
        if not self.root:
            return None
        return(self.root.value)

def test_queue():
    q = Queue()
    q.push(1)
    q.push(2)
    q.push(3)
    assert q.pop() == 1
    assert q.pop() == 2
    assert q.pop() == 3
    print("test_queue() PASSED")

def test():
    a = [50]
    m = MinHeap(a)
    assert m.peak() == 50

    m.push(30)
    assert m.peak() == 30

    m.push(80)
    assert m.peak() == 30

    m.push(10)
    print(m.peak())

    # 50, 30
    # 50 -> 30
    assert m.peak() == 10

    assert m.pop() == 10
    assert m.peak() == 30

    print("test() PASSED")

def test_2():
    assert m.pop() == 4
    assert m.pop() == 7

    from random import random 
    a = [int(random() * 100) for _ in range(200)]
    # random index pop
    # check min
    # random value push
    # check min

if __name__ == "__main__":
    #test_queue()
    test()
