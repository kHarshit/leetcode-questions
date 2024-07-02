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
    class Solution:
		def partitionString(self, s: str) -> int:
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
6. [704. Binary Search](https://leetcode.com/problems/binary-search/description/): Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1. You must write an algorithm with O(log n) runtime complexity.
	```python
	class Solution:
		def search(self, nums: List[int], target: int) -> int:
			low, high = 0, len(nums) - 1

			while low <= high:
				mid = low + (high - low) // 2
				if nums[mid] == target:
					return mid
				elif nums[mid] < target:
					# check in right half
					low = mid + 1
				elif nums[mid] > target:
					# check in left half
					high = mid - 1
			
			return -1
			
		#     return self.binary_search(nums, 0, len(nums)-1, target)

		# def binary_search(self, nums: List[int], low:int, high: int, target: int) -> int:

		#     if high >= low:
		#         mid = low + (high-low)//2

		#         if nums[mid] == target:
		#             return mid
		#         elif nums[mid] < target:
		#             return self.binary_search(nums, mid+1, high, target)
		#         elif nums[mid] > target:
		#             return self.binary_search(nums, low, mid-1, target)

		#     else:
		#         return -1

	# 1, 2, 4, 5, 5, 6, 7, 8
	# target = 5
	```
7. [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/): Given a string s, find the length of the longest substring without repeating characters.
	```python
	class Solution:
		# def lengthOfLongestSubstring(self, s: str) -> int:
		#     max_length = 0
		#     left  = 0
		#     substring = set()
		#     # increment right pointer
		#     for right, val in enumerate(s):
		#         # remove characters from set until duplicates removed
		#         while val in substring:
		#             substring.remove(s[left])
		#             left += 1
		#         substring.add(val)
		#         max_length = max(max_length, right - left + 1)
		#     return max_length
		
		def lengthOfLongestSubstring(self, s: str) -> int:
			max_len = 0
			substring = deque()
			for idx, val in enumerate(s):
				while val in substring:
					substring.popleft()
				substring.append(val)
				max_len = max(max_len, len(substring))

			return max_len
	```
8. [110. Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/): Given a binary tree, determine if it is height-balanced (depth of the two subtrees of every node never differs by more than one.).
	```python
	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution:
		def isBalanced(self, root: Optional[TreeNode]) -> bool:
			
			def dfs(root):
				if not root:
					# balanced, height
					return [True, 0]
				# call dfs on both left and right subtree
				left, right = dfs(root.left), dfs(root.right)
				# differenece in height
				# and left tree balanced and right tree balanced
				isBalanced = left[0] and right[0] and abs(left[1] - right[1]) <= 1
				return [isBalanced, 1 + max(left[1], right[1])]
			
			return dfs(root)[0]
	```
9. [226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/): Given the root of a binary tree, invert the tree, and return its root.
	```python
	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution:
		def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
			if not root:
				return None

			# swap left and right children
			tmp = root.left
			root.left = root.right
			root.right = tmp

			# dfs (recursively call left and right subnodes)
			self.invertTree(root.left)
			self.invertTree(root.right)

			return root
	```
10. [1382. Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/description/): Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them. A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.
	```python
	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution:
		def balanceBST(self, root: TreeNode) -> TreeNode:
			# store inorder traversal
			inorder = []
			self.inorder_traversal(root, inorder)
			return self.create_balanced_bst(inorder, 0, len(inorder)-1)
			
		def inorder_traversal(self, root: TreeNode, inorder: list) -> TreeNode:
			if not root:
				return None
			self.inorder_traversal(root.left, inorder)
			inorder.append(root.val)
			self.inorder_traversal(root.right, inorder)

		def create_balanced_bst(self, inorder: list, start: int, end: int) -> TreeNode:
			# base case
			if start > end: 
				return None

			# mid point
			mid = start + (end - start)//2

			left_subtree = self.create_balanced_bst(inorder, start, mid-1)
			right_subtree = self.create_balanced_bst(inorder, mid+1, end)

			node = TreeNode(inorder[mid], left_subtree, right_subtree)

			return node
	```
11. [105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/): Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.
	```python
	# Definition for a binary tree node.
	# class TreeNode:
	#     def __init__(self, val=0, left=None, right=None):
	#         self.val = val
	#         self.left = left
	#         self.right = right
	class Solution:
		def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
			# base case
			if not preorder or not inorder:
				return None
			# preorder: root, left, right (first element is root)
			# inorder: left, root, right (we split on root, elements left to root are in left subtree, right to root are in right subtree)
			# postorder: left, right, root
			root = TreeNode(preorder[0])
			split_point = inorder.index(preorder[0])
			# for preorder, remove first element and pass preorder list till split_point
			root.left = self.buildTree(preorder[1:split_point+1], inorder[:split_point])
			# for right children, take right-half of arrays
			root.right = self.buildTree(preorder[split_point+1:], inorder[split_point+1:])

			return root
	```