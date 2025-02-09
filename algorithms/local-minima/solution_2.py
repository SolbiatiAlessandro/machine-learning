# ==Example==
# [5, 9, 7, 10, 12]: Output 0 or 2
# [1, 2, 3, 6]: Output 0

def local_minima(a): # [1, 2, 3, 6]
    # binary search
    start, end = 0, len(a) # 4

    while start <= end:  
        # print(start, end)
        m = start  + int((start + end) / 2) # 0
        current_value = a[m] # 1
        right_value = a[m+1] if (m+1) < len(a) else 10e32 # 2
        left_value = a[m-1] if (m-1) >= 0 else 10e32 #  1032
        if current_value <= right_value and current_value <= left_value:
            return m
        if current_value > right_value:
            start = m + 1
        else:
            end = m - 1 # 1

def local_minima_2(nums):
    if len(nums) == 0: return -1 

    left, right = 0, len(nums) - 1

    while left <= right:

        if left == right:
            return left

        mid = left + (right - left) // 2

        if nums[mid] < nums[mid - 1]:
            left = mid
        else:
            right = mid - 1

def test():
    print(local_minima([5, 9, 7, 10, 12]))
    print(local_minima([1, 2, 3, 6]))
    print(local_minima_2([5, 9, 7, 10, 12]))
    print(local_minima_2([1, 2, 3, 6]))

if __name__ == "__main__":
    test()
