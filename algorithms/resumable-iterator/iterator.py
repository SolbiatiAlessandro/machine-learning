STOP_ITERATOR = "stop_iterator"

from dataclasses import dataclass

class IteratorEmpty(Exception):
    pass

class IteratorStateFormatError(Exception):
    pass

class StopIteration(Exception):
    pass

class Iterator:
    def __init__(self, data):
        self.data = data
        self.ix = 0

    def next(self): 
        if self.ix >= len(self.data): raise StopIteration
        res = self.data[self.ix]
        self.ix += 1
        return res

    def get_state(self):
        return (tuple(self.data), self.ix)

    def set_state(self, state):
        assert len(state) == 2
        assert type(state[0]) == tuple
        assert type(state[1]) == int
        self.data = state[0]
        self.ix = state[1]
        


def assert_stop_iteration(iterator):
    try:
        iterator.next()
    except Exception as e:
        assert type(e) == StopIteration

def test():

    # basic test
    iterator = Iterator([1,2,3])
    for aa in [1,2,3]:
        assert iterator.next() == aa
    print("1 TEST PASSED")
    assert_stop_iteration(iterator)

    # stop iteration
    print("1 TEST PASSED")

    # empty iteration
    iterator = Iterator([])
    assert_stop_iteration(iterator)
    print("1 TEST PASSED")

    # resuming test
    iterator = Iterator([1,2,3])
    assert iterator.next() == 1

    state = iterator.get_state()
    assert iterator.next() == 2
    assert iterator.next() == 3
    end_state = iterator.get_state()

    iterator.set_state(state)
    assert iterator.next() == 2

    iterator.set_state(state)
    assert iterator.next() == 2

    iterator.set_state(end_state)
    assert_stop_iteration(iterator)

    # same get state
    s1 = iterator.get_state() 
    s2 = iterator.get_state() 
    assert s1 == s2
    print("1 TEST PASSED")

test()


