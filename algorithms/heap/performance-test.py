from heaparray import MinHeap
import heapq
import random
import time

#--------------------------------------------------------------------
# Performance Comparison Test
def compare_performance(_size=10000):
    print(f"compare_performance O({_size})")
    size = _size          # Number of values to insert
    check_iterations = _size  # Number of times to check the min repeatedly
    pop_count = size // 2      # Number of pop operations

    # Generate a list of random integers.
    values = [random.randint(0, 10000) for _ in range(size)]

    #---- Test using the custom heap ----
    heap = MinHeap([])  # start with an empty heap

    start_time = time.time()
    # Insertion phase for the heap.
    for val in values:
        heap.push(val)
    heap_insertion_time = time.time() - start_time

    start_time = time.time()
    # Repeatedly check the minimum.
    for _ in range(check_iterations):
        _ = heap.peak()
    heap_peak_time = time.time() - start_time

    start_time = time.time()
    # Pop a bunch of values.
    for _ in range(pop_count):
        heap.pop()
    heap_pop_time = time.time() - start_time

    heap_total_time = heap_insertion_time + heap_peak_time + heap_pop_time

    print("Custom Heap Performance:")
    print("  Insertion time: {:.6f} s".format(heap_insertion_time))
    print("  Peak check time: {:.6f} s".format(heap_peak_time))
    print("  Pop time:        {:.6f} s".format(heap_pop_time))
    print("  Total time:      {:.6f} s".format(heap_total_time))
    print("-" * 50)

    #---- Test using a plain unsorted array ----
    arr = []
    start_time = time.time()
    # Insertion phase for the array (just append at the end).
    for val in values:
        arr.append(val)
    array_insertion_time = time.time() - start_time

    start_time = time.time()
    # Repeatedly check the minimum by calling min(arr).
    for _ in range(check_iterations):
        _ = min(arr)
    array_peak_time = time.time() - start_time

    start_time = time.time()
    # Pop a bunch of values by finding and removing the min.
    for _ in range(pop_count):
        m = min(arr)
        arr.remove(m)
    array_pop_time = time.time() - start_time

    array_total_time = array_insertion_time + array_peak_time + array_pop_time

    print("Plain Unsorted Array Performance:")
    print("  Insertion time: {:.6f} s".format(array_insertion_time))
    print("  Peak check time: {:.6f} s".format(array_peak_time))
    print("  Pop time:        {:.6f} s".format(array_pop_time))
    print("  Total time:      {:.6f} s".format(array_total_time))
    print("=" * 50)


     #---- Test 3: Python's Standard Heap Library (heapq) ----
    py_heap = []
    start_time = time.time()
    # Insertion phase using heapq.heappush.
    for val in values:
        heapq.heappush(py_heap, val)
    heapq_insertion_time = time.time() - start_time

    start_time = time.time()
    # Repeatedly check the minimum (heap[0]).
    for _ in range(check_iterations):
        _ = py_heap[0]
    heapq_peak_time = time.time() - start_time

    start_time = time.time()
    # Pop a bunch of values using heapq.heappop.
    for _ in range(pop_count):
        heapq.heappop(py_heap)
    heapq_pop_time = time.time() - start_time

    heapq_total_time = heapq_insertion_time + heapq_peak_time + heapq_pop_time

    print("Python heapq Performance:")
    print("  Insertion time: {:.6f} s".format(heapq_insertion_time))
    print("  Peak check time: {:.6f} s".format(heapq_peak_time))
    print("  Pop time:        {:.6f} s".format(heapq_pop_time))
    print("  Total time:      {:.6f} s".format(heapq_total_time))
    print("=" * 50)


if __name__ == '__main__':
    compare_performance(_size=10000)
    compare_performance(_size=20000)
