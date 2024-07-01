# leetcode-questions
Leetcode questions

1. [1492 The kth Factor of n](https://leetcode.com/problems/the-kth-factor-of-n/description/): You are given two positive integers n and k. A factor of an integer n is defined as an integer i where n % i == 0. Consider a list of all factors of n sorted in ascending order, return the kth factor in this list or return -1 if n has less than k factors.
   ```python
   class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        # O(n)
        factor_num = 0
        for j in range(1, n+1):
            if n % j == 0:
                factor_num += 1
                if factor_num == k:
                    return j
                
        return -1
   ```
2. [2405. Optimal Partition of String](https://leetcode.com/problems/optimal-partition-of-string/description/): Given a string s, partition the string into one or more substrings such that the characters in each substring are unique. That is, no letter appears in a single substring more than once. Return the minimum number of substrings in such a partition. Note that each character should belong to exactly one substring in a partition.
   ```python
           # Maintain only count
        substring = []
        substring_count = 0
        for i in s:
            if i in substring:
                substring = [i]
                substring_count += 1 
            else:
                if not substring:
                    substring_count += 1
                substring.append(i)

        # if substring isn't empty, account for last partition
        if not substring:
            substring_count += 1 

        return substring_count
   ```
3. [1. Two Sum](https://leetcode.com/problems/two-sum/description/): Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.
   ```python
   class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # O(n^2)
        # iterate all 0 to n
        # for i in range(len(nums)):
        #     # itearte from i+1 to n
        #     for j in range(i+1, len(nums)):
        #         # check sum
        #         if nums[i] + nums[j] == target:
        #             return [i, j]

        # 
        
        # dict{val} = idx
        dict_nums = {}
        for idx, val in enumerate(nums):
            diff = target - val
            if diff in dict_nums:
                return [idx, dict_nums[diff]]
            dict_nums[val] = idx
   ```
4. [167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/) Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length. Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2. The tests are generated such that there is exactly one solution. You may not use the same element twice. Your solution must use only constant extra space.
   ```python
   class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        # maintain two pointers
        left, right = 0, len(numbers)-1
        for idx, val in enumerate(numbers):
            sum = numbers[left] + numbers[right]
            if sum == target:
                return [left+1, right+1]
            # if sum > target, decrease right pointer as array is already sorted
            # and sum is higher, so we need to take into account lower numbers
            elif sum > target:
                right -= 1
            elif sum < target:
                left += 1
        return []
   ```
5. [15. 3Sum](https://leetcode.com/problems/3sum/description/): Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Notice that the solution set must not contain duplicate triplets.
   ```python
       def threeSum(self, nums: List[int]) -> List[List[int]]:
        # sort to remove duplicate entries
        nums.sort()
        target = 0
        output_lists = []
        for idx, val in enumerate(nums):
            # for non-first value of arrays:
            # no need to check same values e.g. [1, 3, 3, ...., 5]
            # as we don't want duplicates
            if idx > 0 and val == nums[idx-1]:
                continue
            # perform two sum with two pointers
            left, right = idx + 1, len(nums)-1

            # move left pointer towards right, and left pointer towards left
            # until they both come together
            while left < right:
                sum = val + nums[left] + nums[right]
                if sum > 0:
                    right -= 1
                elif sum < 0:
                    left += 1
                elif sum == 0:
                    output_lists.append([val, nums[left], nums[right]])
                    left += 1
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1

        return output_lists
   ```
