import time
import random
from collections import defaultdict

def test_list(n, num_ops):
    # Preallocate a list of size n with zeros.
    lst = [0] * n
    start = time.perf_counter()
    for _ in range(num_ops):
        idx = random.randrange(n)
        lst[idx] = lst[idx] + 1  # update the element
        _ = lst[idx]             # read the element
    end = time.perf_counter()
    return end - start

def test_defaultdict(n, num_ops):
    # Use defaultdict with int, which defaults to 0.
    dd = defaultdict(int)
    start = time.perf_counter()
    for _ in range(num_ops):
        idx = random.randrange(n)
        dd[idx] = dd[idx] + 1     # update the value for key idx
        _ = dd[idx]               # read the value
    end = time.perf_counter()
    return end - start

def main():
    n = 100_000          # list size and key range for defaultdict
    num_ops = 20 * n     # total operations (10 * n = 1,000,000 iterations)

    # Set the random seed for reproducibility.
    random.seed(42)

    # Time the list operations.
    list_time = test_list(n, num_ops)
    # Time the defaultdict operations.
    dd_time = test_defaultdict(n, num_ops)

    print(f"List time: {list_time:.6f} seconds")
    print(f"Defaultdict time: {dd_time:.6f} seconds")

if __name__ == '__main__':
    main()

