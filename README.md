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

12. [509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/description/): The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is, F(0) = 0, F(1) = 1; F(n) = F(n - 1) + F(n - 2), for n > 1. Given n, calculate F(n).
	```python
	class Solution:
		def fib(self, n: int) -> int:
			# 1. recursion
			# time: O(2^n), space: O(n)
			if n == 0:
				return 0
			elif n == 1:
				return 1
			else:
				return self.fib(n-1) + self.fib(n-2)

			# 2. DP Tabulation
			# time: O(n), space: O(n)
			if n == 0:
				return 0
			elif n == 1:
				return 1
			else:
				# store a list of fibonacci numbers
				# fib_nums = [0] * (n+1) # 1 extra to handle n = 0
				# fib_nums[0] = 0
				# fib_nums[1] = 1
				# for i in range(2, n+1):
				#     fib_nums[i] = fib_nums[i-1] + fib_nums[n-2]

				# return fib_nums[n]
			
				# you don't even need to store all numbers, just last two
				# time: O(n), space: O(1)
				num1, num2 = 0, 1
				for i in range(2, n+1):
					sum = num1 + num2
					num1 = num2
					num2 = sum
				
				return num2
	```

13. 0-1 Knapsack problem: Given N items where each item has some weight and profit associated with it and also given a bag with capacity W, [i.e., the bag can hold at most W weight in it]. The task is to put the items into the bag such that the sum of profits associated with them is the maximum possible. 
	```python
	def knapsack(weights, values, capacity):
		# time: O(n*capacity), space: O(n*capacity)
		n = len(weights)
		# create 2d array: (nx1)x(capacity+1)
		dp = []
		for i in range(n+1):
			dp.append([0] * (capacity+1))

		# iterate through table
		for i in range(n+1):
			for w in range(capacity + 1):
				if i == 0 or w == 0:
					dp[i][w] = 0
				# if item can be included:
				# take max of including the item and not including it
				elif weights[i-1] <= w:
					dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], d[i-1][w])
				# if item can't be included i.e. its weight is greater than capacity, we carry forward the value without including the item. 
				else:
					dp[i][w] = dp[i-1][w]

		return d[n][capacity]
		# Example usage:
		weights = [1, 2, 3, 4]
		values = [1, 4, 5, 7]
		capacity = 7
		print("Maximum value in Knapsack =", knapsack(weights, values, capacity))
	
14. [322. Coin Change](https://leetcode.com/problems/coin-change/description/): You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1. You may assume that you have an infinite number of each kind of coin.
	```python
	class Solution:
		def coinChange(self, coins: List[int], amount: int) -> int:
			# Initialize dp array where dp[i] represents the minimum coins needed for amount i
			dp = [float('inf')] * (amount+1) # for amounts: 0...amount

			# base case: for amount 0, we need 0 coins
			dp[0] = 0

			# for each amount, check each coin
			for i in range(1, amount+1):
				# If the coin can be used (i.e., the remaining amount i - coin is non-negative), we update the dp value to the minimum number of coins needed.
				for coin in coins:
					if i - coin >= 0:
						dp[i] = min(dp[i], 1 + dp[i-coin])
			# if d[amount] is still inf -> not possible to form amount with given coins
			return dp[amount] if dp[amount] != float('inf') else -1
	```

15. [518. Coin Change II](https://leetcode.com/problems/coin-change-ii/description/): You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0. You may assume that you have an infinite number of each kind of coin. The answer is guaranteed to fit into a signed 32-bit integer.
	```python
	class Solution:
		def change(self, amount: int, coins: List[int]) -> int:
			# 1. using 2d arrays
			n = len(coins)
			dp = []
			for i in range(n+1):
				dp.append([0] * (amount+1))

			# Base case initialization: There's 1 way to make up amount 0 with any number of coins i.e. no coins
			for i in range(n + 1):
				dp[i][0] = 1

			# iterate through table from 1 to n+1 (skip first row)
			# In 2d table, rows: coins, columns: sum
			for i in range(1, n+1):
				for j in range(amount+1):

					# if we exclude the current coin, take previous row value
					# Add the number of ways to make change without using the current coin
					dp[i][j] = dp[i-1][j]

					# if we use the current coin
					if j >= coins[i-1]:
						dp[i][j] += dp[i][j-coins[i-1]]

			return dp[n][amount]

			# 2. using 1d array, where each row keeps replacing previous row
			# # Initialize dp array where dp[i] represents the number of ways to make amount i
			# dp = [0] * (amount + 1)

			# # Base case: There's one way to make amount 0 (no coins)
			# dp[0] = 1  

			# # For each coin in the given list of coins, iterate through all amounts from the coin's value to the total amount.
			# for coin in coins:
			#     for i in range(coin, amount + 1):
			#         dp[i] += dp[i - coin]

			# return dp[amount]
	```