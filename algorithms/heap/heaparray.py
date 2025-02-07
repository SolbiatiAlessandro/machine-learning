class MinHeap:
    def __init__(self, values, debug=False):
        self._debug = debug
        self.array = ['EMPTY']
        for v in values:
            self.push(v)

    def get_children_index(self, i):
        if i == 0:
            raise AssertionError("self.array indexed in 1")
        return [2*i, 2*i + 1]

    def get_parent_index(self, i):
        if i == 1:
            return None
        return int(i / 2)

    def push(self, value):
        self.array.append(value)
        # $, 30, 50, 80, 10

        current_index = len(self.array) - 1    # 4
        parent_index = self.get_parent_index(current_index)  # 2
        while parent_index and self.array[parent_index] > self.array[current_index]:
            temp = self.array[parent_index] # 50
            self.array[parent_index] = self.array[current_index] # 30, 10, 80, 10
            self.array[current_index] = temp #30, 10, 80, 50
            current_index =  parent_index #2
            parent_index = self.get_parent_index(current_index)  # 2

        if self._debug: print("push", value, self.array)


    def pop(self):
        # ['EMPTY', 10, 30, 80, 50, 40, 100]
        res = self.array[1] # 10

        self.array[1] = self.array[-1] # $, 100, 30, 80, 50, 40, 100
        self.array = self.array[:-1] # $, 100, 30, 80, 50, 40

        swap_index = 1
        while swap_index:
            current_index = swap_index # 5
            current_value = self.array[current_index] # 100
            left_index, right_index = self.get_children_index(current_index) # 4, 5
            left_children = self.array[left_index] if len(self.array) >  left_index else None # 50
            right_children = self.array[right_index] if len(self.array) >  right_index else None # 40

            swap_index = None
            if left_children is None:
                swap_index = None
            elif right_children is None or left_children < right_children: 
                if left_children < current_value: # 30 < 50
                    swap_index = left_index # 2
            elif right_children <= left_children:
                if right_children < current_value:
                    swap_index = right_index # 5

            if swap_index: # 5
                temp = self.array[swap_index] # 40
                self.array[swap_index] = self.array[current_index] # $, 100, 100, 50, 40, 80
                self.array[current_index] = temp # $, 30, 100, 50, 40, 80

        if self._debug: print("pop", self.array)
        return res

    def peak(self):
        return self.array[1] if len(self.array) > 1  else None

def test_heap_deterministic():
    a = [50]
    m = MinHeap(a)
    assert m.peak() == 50

    m.push(30)
    assert m.peak() == 30

    m.push(80)
    assert m.peak() == 30

    m.push(10)
    assert m.peak() == 10

    assert m.pop() == 10
    assert m.peak() == 30

    m.push(10)
    m.push(40)
    m.push(100)
    assert m.pop() == 10
    assert m.pop() == 30
    assert m.pop() == 40

    print("test() PASSED")

from random import random, randint

def test_heap_randomized():
    size = int(random() * 10000)

    # Create a list of 200 random integers in the range [0, 100)
    a = [int(random() * size) for _ in range(size)]
    print("Initial array:")
    print(len(a))

    # Build the heap from the list
    m = MinHeap(a)


    pop_test_length = int(random() * size)
    for _ in range(pop_test_length):
        # Check that the current min equals the minimum of the array.
        expected_min = min(a)
        actual_min = m.peak()
        assert actual_min == expected_min, f"Expected min {expected_min}, but got {actual_min}"

        # Pop the minimum element from the heap
        popped = m.pop()  # This should remove the root and reheapify.
        # print(f"\nPopped value: {popped}")
        # Remove the popped value from the original array copy
        a.remove(popped)

        # Check that the new heap minimum equals the new minimum in the array (if any)
        if a:  # if the heap is not empty
            expected_min = min(a)
            actual_min = m.peak()
            # print(f"TEST pop(): After pop, expected min: {expected_min}, heap.peak(): {actual_min}")
            assert actual_min == expected_min, f"Expected min {expected_min}, but got {actual_min}"
        else:
            #print("Heap is empty after pop.")
            pass
    print(f"TEST pop(): {pop_test_length} TESTS PASSED")


    push_test_length = int(random() * size)
    for _ in range(push_test_length):
        # Push a new random value into the heap
        new_value = int(random() * 100)
        #print(f"\nPushing new value: {new_value}")
        m.push(new_value)
        a.append(new_value)

        # Check that the heap's min matches the min of the updated list.
        expected_min = min(a)
        actual_min = m.peak()
        #print(f"TEST push(): After push, expected min: {expected_min}, heap.peak(): {actual_min}")
        assert actual_min == expected_min, f"Expected min {expected_min}, but got {actual_min}"

    print(f"TEST push(): {push_test_length} TESTS PASSED")

if __name__ == "__main__":
    test_heap_deterministic()
    for _ in range(20):
        test_heap_randomized()
