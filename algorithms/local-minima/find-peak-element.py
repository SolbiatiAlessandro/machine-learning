# https://leetcode.com/problems/find-peak-element/

# accepted, beats 75

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:

        start, end = 0, len(nums)
        while True:
            m = start + int((end - start) / 2)
            curr = nums[m]
            left = nums[m - 1] if m > 0 else -2e33
            right = nums[m + 1] if (m + 1) < len(nums) else -2e33
            if right > curr:
                start = m + 1
            elif left > curr:
                end = m - 1
            else:
                return m


# [1,2,3,1] 
# [1,2,1,3,5,6,4]

#[1,2,3,4,5,6,7,6,5]
        
