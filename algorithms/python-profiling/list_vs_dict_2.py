import time
import random
from collections import defaultdict

def test_list(n, num_ops):
    # Allocate a fixed-size list of 100_000 elements.
    lst = [0] * 100_000  
    start = time.perf_counter()
    for _ in range(num_ops):
        idx = random.randrange(100_000)  # use only indices in range [0, n)
        # Increment the element and then read it.
        lst[idx] += 1
        _ = lst[idx]
    end = time.perf_counter()
    return end - start

def test_dict(n, num_ops):
    # Prepopulate a defaultdict with keys 0..n-1 (each starting at 0).
    dd = defaultdict(int)
    for i in range(n):
        dd[i] = 0
    start = time.perf_counter()
    for _ in range(num_ops):
        idx = random.randrange(100_000)  # keys only between 0 and n-1
        dd[idx] += 1
        _ = dd[idx]
    end = time.perf_counter()
    return end - start

def main():
    iterations = 20
    total_list_time = 0.0
    total_dict_time = 0.0

    # For reproducibility.
    random.seed(42)

    for _ in range(iterations):
        # Choose n uniformly between 10 and 100,000.
        n = random.randint(10, 100_000)
        num_ops = 100000

        lt = test_list(n, num_ops)
        dt = test_dict(n, num_ops)

        total_list_time += lt
        total_dict_time += dt

    avg_list_time = total_list_time / iterations
    avg_dict_time = total_dict_time / iterations

    print(f"Average list time: {avg_list_time:.6f} seconds")
    print(f"Average defaultdict time: {avg_dict_time:.6f} seconds")

if __name__ == '__main__':
    main()

