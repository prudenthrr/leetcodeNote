### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

**题目描述：**

给出两个 **非空** 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 **逆序** 的方式存储的，并且它们的每个节点只能存储 **一位** 数字。

如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。

您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```



**解题思路：**

使用一个变量carry来保存进位数，对两个链表同时进行遍历，取出两个链表的当前元素并与carry求和sum，修改carry的值供下次计算使用，存储当前sum的个位数。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        head = ListNode(0)
        ptr = head
        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            s = carry + x + y
            carry = s // 10     # 十位数赋值给进位数
            ptr.next = ListNode(s % 10)   # 存储个位数
            ptr = ptr.next
            if l1: l1 = l1.next
            if l2: l2 = l2.next
        if carry==1:
            ptr.next = ListNode(1)
        return head.next
```



### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)（重点）

**题目描述**：

给定一个**非空**字符串 *s* 和一个包含**非空**单词列表的字典 *wordDict*，判定 *s* 是否可以被空格拆分为一个或多个在字典中出现的单词。

**说明：**

- 拆分时可以重复使用字典中的单词。
- 你可以假设字典中没有重复的单词。

**示例 1：**

```
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
```

**示例 2：**

```
输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
```

**示例 3：**

```
输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false
```

 

 **解题思路：**

使用动态规划。使用dp 来存储是否匹配到字典中的单词的bool值。dp得长度为字符串长度加1，初试状态设置为True。dp[i]表示长度为i的字符串。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        size = len(s)
        # dp[i]：长度为 i 的 s 字符串经过空格分隔以后在 wordDict 中
        # 需要长度为 0，因此前面加上一个 True
        dp = [True] + [False for _ in range(size)]
        dic = {}
        for i in wordDict:
            dic[i] = 1
        for i in range(1, size + 1):
            # j 其实就是前缀的长度，注意这个细节
            for j in range(i):
                right = s[j:i]
                if right in dic and dp[j]:
                    dp[i] = True
                    # 注意，这里 break
                    break
        return dp[-1]

```





### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

**题目描述**：

给定一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过**水平方向或垂直方向**上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

**示例 1:**

```
输入:
11110
11010
11000
00000

输出: 1
```

**示例 2:**

```
输入:
11000
11000
00100
00011

输出: 3
```



**解题思路：**

使用深度优先遍历算法。首先定位一个值为''1''的位置，并将其值设置为”2“，然后从该位置开始查看上下左右四个方向的值。如果查看到的值为”1“，则将该位置上的值设置为”2“，接着再从四个方向进行查看。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # 判断边界条件
        if len(grid)==0 or len(grid[0])==0:
            return 0
		
        def helper(i,j):
            grid[i][j] = '2'
            # 向上下左右走，注意边界条件
            if i-1>=0 and grid[i-1][j]=='1':
                helper(i-1, j)
            if i+1<len(grid) and grid[i+1][j]=='1':
                helper(i+1, j)
            if j-1>=0 and grid[i][j-1]=='1':
                helper(i, j-1)
            if j+1<len(grid[0]) and grid[i][j+1]=='1':
                helper(i, j+1)

        c = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    c += 1
                    helper(i,j)
        return c
```



### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

题目描述：

编写一个高效的算法来搜索 *m* x *n* 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

**示例:**

现有矩阵 matrix 如下：

```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

给定 target = `5`，返回 `true`。

给定 target = `20`，返回 `false`。



**解题思路：**

解决方案：先判断行，如果target在行内，则使用二分查找法进行查找。如果没有查找到，则判断列，如果target在行内，则使用二分查找法进行查找。如果行内和列内都找不到，在将行和列分别加1。即从对角线为起始点一层一层超找

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix)==0 or len(matrix[0])==0:
            return False
        m, n = len(matrix), len(matrix[0])
        i, j = 0, 0
        while i<m and j < n:
            # 判断行
            if matrix[i][j]<=target<=matrix[i][n-1]:
                s, e = j, n
                while e-s>1:
                    mid = ( s + e ) // 2
                    if matrix[i][mid]==target:
                        return True
                    elif matrix[i][mid]<target:
                        s = mid
                    else:
                        e = mid
                else:
                    if matrix[i][s]==target or matrix[i][e]==target:
                        return True
            if matrix[i][j]<=target<=matrix[m-1][j]:
                s, e = i, m
                while e-s>1:
                    mid = ( s + e ) // 2
                    if matrix[mid][j]==target:
                        return True
                    elif matrix[mid][j]<target:
                        s = mid
                    else:
                        e = mid
                else:
                    if matrix[s][j]==target or matrix[e][j]==target:
                        return True
            i += 1
            j += 1
        return False
```





### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

**题目描述：与322类似**

给定正整数 *n*，找到若干个完全平方数（比如 `1, 4, 9, 16, ...`）使得它们的和等于 *n*。你需要让组成和的完全平方数的个数最少。

**示例 1:**

```
输入: n = 12
输出: 3 
解释: 12 = 4 + 4 + 4.
```

**示例 2:**

```
输入: n = 13
输出: 2
解释: 13 = 4 + 9.
```

**解题思路：**

使用动态规划。状态转移公式为： dp[i] = min(dp[i], dp[i-j*j]+1)

```python 
class Solution:
    def numSquares(self, n: int) -> int:
        import math
        dp = [float("inf")]*(n+1)
        dp[0] = 0
        for i in range(1,n+1):
            for j in range(int(math.sqrt(i)), 0, -1):
                dp[i] = min(dp[i], dp[i-j*j]+1)  
        return dp[-1]
```



### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)（不太理解）

**题目描述：**

给定一个包含 *n* + 1 个整数的数组 *nums*，其数字都在 1 到 *n* 之间（包括 1 和 *n*），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

**示例 1:**

```
输入: [1,3,4,2,2]
输出: 2
```

**示例 2:**

```
输入: [3,1,3,4,2]
输出: 3
```

**说明：**

1. **不能**更改原数组（假设数组是只读的）。
2. 只能使用额外的 *O*(1) 的空间。
3. 时间复杂度小于 *O*(*n*2) 。
4. 数组中只有一个重复的数字，但它可能不止重复出现一次。

**解题思路：**

快慢指针思想, fast 和 slow 是指针, nums[slow] 表示取指针对应的元素
注意 nums 数组中的数字都是在 1 到 n 之间的(在数组中进行游走不会越界),
因为有重复数字的出现, 所以这个游走必然是成环的, 环的入口就是重复的元素, 
即按照寻找链表环入口的思路来做

**解法一（重点）**

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        nums_sum = sum(nums)
        slow = 0
        fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        finder = 0
        while True:
            slow = nums[slow]
            finder = nums[finder]
            if slow == finder:
                return slow
```

解法二：

```python
class Solution:
    def findDuplicate(self, nums) -> int:
        nums_sum = sum(nums)
        set_sum = sum(set(nums))
        return (nums_sum - set_sum)//(len(nums)-len(set(nums)))
```



### [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)(重点）

**题目描述：**

给定一个无序的整数数组，找到其中最长上升子序列的长度。

**示例:**

```
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
```

**说明:**

- 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
- 你算法的时间复杂度应该为 O(*n2*) 。

**进阶:** 你能将算法的时间复杂度降低到 O(*n* log *n*) 吗?



**解题思路：**

使用动态规划。状态转移函数：dp[i] = max(dp[i], dp[j+1])

```python 
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n==0:
            return 0
        dp = [1]*n
        for i in range(n):
            for j in range(i):
                # 核心代码
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```



### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)（不太理解）

**题目描述：**

给定一个整数数组，其中第 *i* 个元素代表了第 *i* 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

- 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
- 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

**示例:**

```
输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```



**解题思路：**

动态规划

sell[i]表示截至第i天，最后一个操作是卖时的最大收益；
buy[i]表示截至第i天，最后一个操作是买时的最大收益；
cool[i]表示截至第i天，最后一个操作是冷冻期时的最大收益；
递推公式：
sell[i] = max(buy[i-1]+prices[i], sell[i-1]) (第一项表示第i天卖出，第二项表示第i天冷冻)
buy[i] = max(cool[i-1]-prices[i], buy[i-1]) （第一项表示第i天买进，第二项表示第i天冷冻）
cool[i] = max(sell[i-1], buy[i-1], cool[i-1])

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0     
        sell = [0]*n
        buy = [0]*n
        cool = [0]*n
        buy[0] = -prices[0]
        for i in range(1,n):
            sell[i] = max(buy[i-1] + prices[i], sell[i-1])
            buy[i] = max(cool[i-1] - prices[i], buy[i-1])
            cool[i] = max(sell[i-1], buy[i-1],cool[i-1])
        return sell[-1]
```



### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)（不太理解）



**题目描述：与279有点类似**

给定不同面额的硬币 coins 和一个总金额 amount。**编写一个函数来计算可以凑成总金额所需的最少的硬币个数**。如果没有任何一种硬币组合能组成总金额，返回 `-1`。

**示例 1:**

```
输入: coins = [1, 2, 5], amount = 11
输出: 3 
解释: 11 = 5 + 5 + 1
```

**示例 2:**

```
输入: coins = [2], amount = 3
输出: -1
```

**说明**:
你可以认为每种硬币的数量是无限的。



**解题思路：**

使用动态规划，本题类似0/1背包问题。首先,0/1背包的动态规划,就是取与不取的问题,dp[i]表示金额为i需要最少的金额多少,对于任意金额j,dp[j] = min(dp[j],dp[j-coin]+1),如果j-coin存在的话.

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float("inf") else -1
```



### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)（重点）

**题目描述：**

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

**示例 1:**

```
输入: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

输出: 7 
解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
```

**示例 2:**

```
输入: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

输出: 9
解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
```



**解题思路：**

使用深度优先遍历。为每个节点定义一个二维数组[偷， 不偷]，

robValue(偷到) = 本节点的值 + 左孩子不偷 + 右孩子不偷

skipValue(不偷) = max(左孩子[偷]，左孩子[不偷]) +max(右孩子[偷]，右孩子[不偷])

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root: TreeNode) -> int:
        if root==None:
            return 0
        def helper(root):
            if root==None:
                return [0,0]
            left = helper(root.left)
            right = helper(root.right)
            rob = root.val + left[1] + right[1]  # 偷
            skip = max(left) + max(right)    # 不偷
            return [rob, skip]
        res = helper(root)
        return max(res)
```



### [347. 前K个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

**题目描述：**

给定一个**非空**的整数数组，返回其中出现频率前 **k** 高的元素。

**示例 1:**

```
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
```

**示例 2:**

```
输入: nums = [1], k = 1
输出: [1]
```

**说明：**

- 你可以假设给定的 *k* 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
- 你的算法的时间复杂度**必须**优于 O(*n* log *n*) , *n* 是数组的大小。



**解题思路：**

使用哈希算法和堆数据结构。使用哈希来存储每个数字出现的频率，再使用堆结构找出前K个频率最高的数字。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        d = {}
        for v in nums:
            if v in d:
                d[v] += 1
            else:
                d[v] = 1
        d = [(v,k) for k,v in d.items()]
        res = [x[1] for x in heapq.nlargest(k,d)]   # 堆数据结构
        return res
```



### [394. 字符串解码](https://leetcode-cn.com/problems/decode-string/)

**题目描述：**

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 *encoded_string* 正好重复 *k* 次。注意 *k* 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 *k* ，例如不会出现像 `3a` 或 `2[4]` 的输入。

**示例:**

```
s = "3[a]2[bc]", 返回 "aaabcbc".
s = "3[a2[c]]", 返回 "accaccacc".
s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".
```



**解题思路：**

使用栈的思想解决该问题。**使用一个栈count_stack来存储数字，使用str_stack来存储相连的字符串**。遍历给定的字符串，当前字符若为 **'['** 时，将数字和字符串压入栈，当前字符若为']'时，将count_stack最后一个位置的数字弹出，将数字、当前字符串以及str_stack[-1]组合成新的字符串，并弹出str_stack[-1]。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        if len(s)==0: return ''
        count_stack,str_stack = [],[]
        tmp_num,  tmp_str = '', ''
        for i in range(len(s)):
            # 合并数字字符
            if '0'<=s[i]<='9':
                tmp_num += s[i]
            elif s[i]=='[':
                if tmp_num!='':
                    count_stack.append(int(tmp_num))
                else:
                    count_stack.append(0)
                tmp_num = ''
                str_stack.append(tmp_str)
                tmp_str = ''
            # 合并字母字符
            elif 'a'<=s[i]<='z' or 'A'<= s[i]<='Z':
                tmp_str += s[i]
            else:
                count = count_stack.pop()
                tmp_str = str_stack[-1] + tmp_str*count
                str_stack.pop()
        return tmp_str
```



### [406. 根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

**题目描述：**  贪心算法

假设有打乱顺序的一群人站成一个队列。 每个人由一个整数对(h, k)表示，其中h是这个人的身高，k是排在这个人前面且身高大于或等于h的人数。 编写一个算法来重建这个队列。

注意：
总人数少于1100人。

示例

```
输入:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

输出:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

  

**解题思路：**

对数组进行重新排序，**按身高降序排列，按人数升序排列**

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]):
        # 按身高降序排列，按人数k升序排列
        num = sorted(people,key=lambda x:(-x[0],x[1]))
        for index,i in enumerate(num):
            if i[1] < index :
                num.insert(i[1],num.pop(index))
        return num
```



### [416. 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

**题目描述：**

给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

注意:

每个数组中的元素不会超过 100
数组的大小不会超过 200
示例 1:

```
输入: [1, 5, 11, 5]
输出: true
解释: 数组可以分割成 [1, 5, 5] 和 [11].
```


示例 2:

```
输入: [1, 2, 3, 5]
输出: false
解释: 数组不能分割成两个元素和相等的子集.
```



**解题思路：**

将问题转换为  寻找目标值为 sum(nums) //2的子集

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        def dfs(nums, target):
            if target<0:
                return
            if target==0:
                return True
            for i in range(len(nums)):
                if i>0 and nums[i]==nums[i-1]:  # 去除重合的元素
                    continue
                else:
                    if dfs(nums[:i]+nums[i+1:], target-nums[i]):
                        return True
            return False

        s = sum(nums)
        if s % 2!=0:   # 如果和是奇数
            return False
        nums.sort()
        target = s // 2
        return dfs(nums,target)
```



### [647. 回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

**题目描述：**

给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。

示例 1:

```
输入: "abc"
输出: 3
解释: 三个回文子串: "a", "b", "c".
```


示例 2:

```
输入: "aaa"
输出: 6
说明: 6个回文子串: "a", "a", "a", "aa", "aa", "aaa".
```



**解题思路：**

使用哈希表存储每个字母出现的位置。当哈希表中已经存在该字符时，定位该字符之前出现的位置，然后判断该区域的字符串是否是回文字符串。

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        if len(s)==0:
            return 0
        res = [1]*len(s)
        dic = {s[0]:list([0])}
        for i in range(1,len(s)):
            res[i] += res[i-1]
            if s[i] in dic:
                for j in dic[s[i]]:
                    if s[j:i + 1] == s[j:i + 1][::-1]:
                        res[i] += 1
                dic[s[i]] = dic[s[i]]+[i]
            else:
                dic[s[i]] = list([i])
        return res[-1]
```





### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

**题目描述：**

根据每日 气温 列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高的天数。如果之后都不会升高，请输入 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的都是 [30, 100] 范围内的整数。



**解题思路：**

使用递减栈，后入栈的元素总比栈顶元素小。

栈顶元素与当前元素做对比：

- 如果当前元素>栈顶元素：弹出栈顶元素，记录两者的下标差值。
- 如果当前元素小于栈顶元素：当前的元素下标入栈

**解法一：递减栈**

```python
class Solution:
    def dailyTemperatures(self, T: List[int]) -> List[int]:
        stack = []
        res = [0]*len(T)
        for index,value in enumerate(T):
            if stack:
                while stack and T[stack[-1]]<value:
                    res[stack[-1]] = index - stack[-1]
                    stack.pop()
            stack.append(index)
        return res
```

解法二：使用暴力解决

两层循环。第一层遍历每个元素，第二层，从当前元素的下一个位置开始遍历。

- 如果第一层的当前元素<第二层的当前元素：记录第二层当前元素的下标

```python
class Solution:
    def dailyTemperatures(self, T):
        if len(T)==0:
            return None
        res = [0]*len(T)
        for i in range(len(T)):
            for j in range(i+1,len(T)):
                if T[j]>T[i]:
                    res[i] = j - i
                    break
        return res
```

### [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

**题目描述：**

数组的每个索引做为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

示例 1:

```
输入: cost = [10, 15, 20]
输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
```


 示例 2:

```
输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出: 6
解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。
```



**解题思路：**

使用变量is_pass来存储在该点上停留，no_pass表示在该点上不停留（跳过该点）。

```
is_pass = min(上个节点的is_pass,上个节点的no_pass ) + 该节点的值
no_pass = 上个节点的is_pass
```

```python
class Solution:
    def minCostClimbingStairs(self, cost) -> int:
        if len(cost)==0 or len(cost)==1:
            return 0
        is_pass, no_pass = cost[0], 0
        for i in range(1, len(cost)):
            tmp = is_pass
            is_pass = min(is_pass, no_pass) + cost[i]
            no_pass = tmp
        return min(is_pass, no_pass)
```

