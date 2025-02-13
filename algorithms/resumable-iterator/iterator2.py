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


class Iterator2D:
    def __init__(self, data2d):
        """
        data2d: A list of lists.
        """
        self.data = data2d

    def next(self):
        """
        Returns the next element in the 2D list in row-major order.
        If the current row is exhausted or empty, it moves to the next row.
        When all rows are exhausted, raise StopIteration.
        """
        # Your implementation here...
        raise NotImplementedError("Iterator2D.next() is not implemented.")

    def get_state(self):
        """
        Returns a snapshot of the current state.
        For example, (data2d, row, col).
        """
        # Consider using a deep copy if data2d is mutable.
        raise NotImplementedError("Iterator2D.get_state() is not implemented.")

    def set_state(self, state):
        """
        Restores the iterator's state from the snapshot.
        Expects state to be in the same format returned by get_state().
        """
        # Validate state format before assignment.
        raise NotImplementedError("Iterator2D.set_state() is not implemented.")



def assert_stop_iteration(iterator):
    try:
        iterator.next()
    except Exception as e:
        assert type(e) == StopIteration


import unittest

# Import your Iterator2D and custom StopIteration from your module.
# For example, if your module is named iterator2d.py, you might do:
# from iterator2d import Iterator2D, StopIteration
# For this snippet, we assume that these names are available in the current namespace.

class TestIterator2D(unittest.TestCase):

    def test_basic_iteration(self):
        """
        Test that the iterator traverses a 2D list in row-major order.
        """
        data2d = [[1, 2, 3], [4, 5], [6]]
        iterator = Iterator2D(data2d)
        expected = [1, 2, 3, 4, 5, 6]
        results = []
        try:
            while True:
                results.append(iterator.next())
        except StopIteration:
            pass
        self.assertEqual(results, expected)

    def test_empty_rows(self):
        """
        Test that the iterator correctly skips empty rows.
        """
        data2d = [[], [10, 20], [], [30], []]
        iterator = Iterator2D(data2d)
        expected = [10, 20, 30]
        results = []
        try:
            while True:
                results.append(iterator.next())
        except StopIteration:
            pass
        self.assertEqual(results, expected)

    def test_exhaustion(self):
        """
        Test that calling next() after all elements are consumed raises StopIteration.
        """
        data2d = [[1], [2]]
        iterator = Iterator2D(data2d)
        self.assertEqual(iterator.next(), 1)
        self.assertEqual(iterator.next(), 2)
        with self.assertRaises(StopIteration):
            iterator.next()

    def test_state_resumption(self):
        """
        Test that saving the state and then restoring it resumes iteration at the correct point.
        """
        data2d = [[1, 2], [3, 4]]
        iterator = Iterator2D(data2d)

        # Consume one element.
        first = iterator.next()
        self.assertEqual(first, 1)

        # Save state after the first element (should be at row 0, col 1).
        saved_state = iterator.get_state()

        # Consume the next element.
        second = iterator.next()
        self.assertEqual(second, 2)

        # Consume another element.
        third = iterator.next()
        self.assertEqual(third, 3)

        # Restore to saved state; we expect the next call to yield the same as the one after saved_state.
        iterator.set_state(saved_state)
        resumed = iterator.next()
        self.assertEqual(resumed, 2)
        self.assertEqual(iterator.next(), 3)
        self.assertEqual(iterator.next(), 4)
        with self.assertRaises(StopIteration):
            iterator.next()

    def test_get_state_idempotence(self):
        """
        Test that multiple calls to get_state() without intervening next() calls return identical snapshots.
        """
        data2d = [[100, 200], [300]]
        iterator = Iterator2D(data2d)
        state1 = iterator.get_state()
        state2 = iterator.get_state()
        self.assertEqual(state1, state2)

    def test_completely_empty(self):
        """
        Test that the iterator immediately raises StopIteration when initialized with an empty 2D list.
        """
        data2d = []
        iterator = Iterator2D(data2d)
        with self.assertRaises(StopIteration):
            iterator.next()

if __name__ == '__main__':
    unittest.main()


