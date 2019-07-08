# LeetCode

## 数组和字符串

### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)（过）

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

### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

**题目描述：**

给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

**解题思路：**

使用双指针来解决。第一个指针start指向不重复字符的起始位置，遍历字符串。将每个字符加入到哈希表中。如果当前的字符在哈希表中，则改变start的位置。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)==0:
            return 0
        start = 0
        maxLen = 0
        dic = {}
        for i in range(len(s)):
            if s[i] in dic:
                start = max(start, dic[s[i]]+1)
            maxLen = max(maxLen, i-start+1)
            dic[s[i]] = i
        return maxLen
```

### [4. 寻找两个有序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)（不懂）

**题目描述**：

给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。你可以假设 nums1 和 nums2 不会同时为空。

示例 1:

```
nums1 = [1, 3]
nums2 = [2]
则中位数是 2.0
```

示例 2:

```
nums1 = [1, 2]
nums2 = [3, 4]
则中位数是 (2 + 3)/2 = 2.5
```

```python
class Solution:
    def findMedianSortedArrays(self, nums1, nums2) -> float:
        m, n = len(nums1), len(nums2)
        # 这里为了保证nums1一定是长度较小的数组
        if m > n:
            nums1, nums2, m, n = nums2, nums1, n, m
        # 题目给定数组不会同时为空，也就是m^2+n^2≠0，由于m≤n，故只要n≠0即可
        if not n:
            raise ValueError("数组长度不同时为零")

        i_min, i_max = 0, m

        # left集合元素数量，如果m+n是奇数，left比right多一个数据
        count_of_left = (m + n + 1) // 2

        while i_min <= i_max:
            i = (i_min + i_max) // 2  # left有i个nums1的元素
            j = count_of_left - i  # left有j个nums2的元素
            if i > 0 and nums1[i - 1] > nums2[j]:
                i_max = i - 1  # i太大，要减少
            elif i < m and nums1[i] < nums2[j - 1]:
                i_min = i + 1  # i太小，要增加
            else:
                if i == 0:
                    max_of_left = nums2[j - 1]
                elif j == 0:
                    max_of_left = nums1[i - 1]
                else:
                    max_of_left = max(nums1[i - 1], nums2[j - 1])

                if (m + n) % 2:
                    return float(max_of_left)  # 结果是浮点数

                if i == m:
                    min_of_right = nums2[j]
                elif j == n:
                    min_of_right = nums1[i]
                else:
                    min_of_right = min(nums1[i], nums2[j])
                return (max_of_left + min_of_right) / 2.0

print(Solution().findMedianSortedArrays([1, 2, 3], [4]))
```



### [8. 字符串转换整数 (atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)（过）

**题目描述：**

请你来实现一个 atoi 函数，使其能将字符串转换成整数。

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

说明：假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，qing返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

示例 1:

```
输入: "42"
输出: 42
示例 2:

输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
```


示例 3:

```
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。

输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。因此无法执行有效的转换。
```


示例 5:

```
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。
```

```python
class Solution:
    def myAtoi(self, str: str) -> int:
        import re
        pattern = r'\s*[-,+]?[0-9]+'
        res = re.match(pattern, str)
        if res:
            if -2**31<=int(res.group())<=2**31-1:
                return int(res.group())
            elif -2**31>int(res.group()):
                return -2**31
            else:
                return 2**31-1
        else:
            return 0
```

### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)（过）

给定 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器，且 n 的值至少为 2。

![img](https://aliyun-lc-upload.oss-cn-hangzhou.aliyuncs.com/aliyun-lc-upload/uploads/2018/07/25/question_11.jpg)

图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

示例:

```
输入: [1,8,6,2,5,4,8,3,7]
输出: 49
```

**解题思路：**

使用双指针。当height[i]<height[j]时， i+=1，否则j -= 1

```
class Solution:
    def maxArea(self, height) -> int:
        re = 0
        if len(height)==0 or len(height)==1:
            return re
        i, j = 0, len(height)-1
        while i<j:
            tmp = min(height[i], height[j])*(j-i)
            re = max(re, tmp)
            if height[i]<height[j]:
                i += 1
            else:
                j -= 1
        return re
```

### [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)（过）

**题目描述：**

罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

```
示例 1:

输入: "III"
输出: 3
示例 2:

输入: "IV"
输出: 4
示例 3:

输入: "IX"
输出: 9
示例 4:

输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
示例 5:

输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_dic = {"I":1 ,"V": 5,  "X": 10, "L": 50,  "C": 100, "D": 500,  "M": 1000,
                      "IV": 4, "IX": 9, "XL": 40,  "XC": 90, "CD": 400, "CM": 900}
        i,re = 0,0
        while i < len(s):   #从左到右检索字符
            if i + 1 < len(s) and s[i] + s[i + 1] in roman_dic:   #相连的两个字符在字典中
                re = re + roman_dic[s[i] + s[i + 1]]
                i += 2
            else:
                re = re + roman_dic[s[i]]
                i += 1
        return re
```

### [14. 最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)（过）

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 ""。

```
示例 1:

输入: ["flower","flow","flight"]
输出: "fl"
示例 2:

输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs)==0:
            return ''
        elif len(strs)==1:
            return strs[0]
        minStr = min(strs)
        maxStr = max(strs)
        for i in range(len(minStr)):
            if minStr[i]!=maxStr[i]:
                return minStr[:i]
        return minStr
```

### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)（过）

**题目描述：**

给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```
例如, 给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```python
class Solution:
    def threeSum(self, nums):
        re = []
        if len(nums)>=3:
            nums.sort()
            for i in range(len(nums)):
                if i != 0 and nums[i] == nums[i - 1]:
                    continue
                else:
                    j, k = i + 1, len(nums) - 1
                    while j < k:
                        tmp = nums[i] + nums[j] + nums[k]
                        if tmp == 0:
                            re.append([nums[i], nums[j], nums[k]])
                            j += 1
                            k -= 1
                            while j < k and nums[j] == nums[j - 1]:
                                j += 1
                            while j < k and nums[k] == nums[k + 1]:
                                k -= 1
                        elif tmp < 0:
                            j += 1
                        else:
                            k -= 1
        return re
```

### [16. 最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

**题目描述：**

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

```
例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
```

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if len(nums)>=3:
            nums.sort()
            close = abs(target-(nums[0] + nums[1] + nums[2]))
            re = nums[0] + nums[1] + nums[2]
            for i in range(len(nums)):
                if i != 0 and nums[i] == nums[i - 1]:
                    continue
                j, k = i + 1, len(nums) - 1
                while j < k:
                    tmp = nums[i] + nums[j] + nums[k]
                    if tmp==target:
                        return target
                    elif tmp < target:
                        j += 1
                    else:
                        k -= 1
                    if close > abs(tmp-target):
                        close = abs(tmp-target)
                        re = tmp
            return re
```

## 链表

### [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)（过）

**题目描述：**

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

```
示例：

给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
```


说明：给定的 n 保证是有效的。

进阶：你能尝试使用一趟扫描实现吗？

**解题思路：**

使用一趟扫描完成。使用一个变量pre来存储倒数第n节点之前的节点，使用N_Node来存储第n个要删除的节点。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if head is None:
            return head
        ptr, pre = head, head
        N_Node, Node_1 = head, None  # 倒数第n个节点和倒数第1个节点
        c = -n  
        while ptr:
            if c==-1 and Node_1 is None:
                Node_1 = ptr
                ptr = ptr.next
                continue
            if Node_1 is not None:
                pre = N_Node
                N_Node = N_Node.next
            ptr = ptr.next
            if c!=-1:
                c += 1
        if N_Node == head:
            return head.next
        else:
            pre.next = N_Node.next
        return head
```



## 动态规划

### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

**题目描述：**

给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：

```
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。示例 2：
```

```
输入: "cbbd"
输出: "bb"
```

**解题思路：**

```
# 基本思路是对任意字符串，如果头和尾相同，那么它的最长回文子串一定是去头去尾之后的部分的最长回文子串加上头和尾。如果头和尾不同，
# 那么它的最长回文子串是去头的部分的最长回文子串和去尾的部分的最长回文子串的较长的那一个。
# P[i,j]P[i,j]表示第i到第j个字符的回文子串数
# dp[i,i]=1
# dp[i,j]=dp[i+1,j−1]+2   |   s[i]=s[j]
# dp[i,j]=max(dp[i+1,j],dp[i,j−1])  |  s[i]!=s[j]
```



```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s)==0:
            return ''
        maxlen = 0
        start = 0  # 回文的位置
        for i in range(len(s)):
            if i- maxlen>=1 and s[i-maxlen-1:i+1]==s[i-maxlen-1:i+1][::-1]:
                start = i-maxlen-1
                maxlen += 2
                continue
            if i - maxlen>=0 and s[i-maxlen:i+1]==s[i-maxlen:i+1][::-1]:
                start = i - maxlen
                maxlen += 1
        return s[start:start+maxlen]

```

### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

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

### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

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

### [300. 最长上升子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)（过）

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

### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

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



## 回溯法

### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)（过）

**题目描述**

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/original_images/17_telephone_keypad.png)



```
示例:

输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
说明:
尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。
```

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        
        def solve(array, solution):
            if len(solution)==len_d:
                re.append(solution)
                return
            for i in range(len(array)):
                for j in range(len(array[i])):
                    newSolution = solution + array[i][j]
                    newArray = array[i+1:]
                    solve(newArray, newSolution)
    
        re = []
        if len(digits) == 0:
            return re
        dic = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
                        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        all = [dic[s] for s in digits]
        len_d = len(digits)
        solve(all, '')
        return re
```

### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

**题目描述：**

给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。

```
例如，给出 n = 3，生成结果为：
[ "((()))","(()())","(())()","()(())","()()()"]
```

```python
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        re = []
        self.helper(re, '', 0, 0, n)
        return re
    def helper(self, re, ans, count1, count2, n):
        if count1==n and count2==n:
            re.append(ans)
            return
        if count1<n:
            self.helper(re, ans+'(', count1+1, count2, n)
        if count1>count2:
            self.helper(re, ans+')',count1, count2+1, n)
```



### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)（过）

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
    def numIslands(self, grid) -> int:
        re = 0
        if len(grid)==0 or len(grid[0])==0:
            return re
		# 核心
        def helper(i, j):
            grid[i][j] = '2'
            if i - 1 >= 0 and grid[i - 1][j] == '1':
                helper(i - 1, j)
            if i + 1 < len(grid) and grid[i + 1][j] == '1':
                helper(i + 1, j)
            if j - 1 >= 0 and grid[i][j - 1] == '1':
                helper(i, j - 1)
            if j + 1 < len(grid[0]) and grid[i][j + 1]== '1':
                helper(i, j + 1)
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    re += 1
                    helper(i, j)
        return re
```



## 查找

### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)（过）

**题目描述：**

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
        j = len(matrix[0])-1
        i = len(matrix)-1
        while i>=0:
            if matrix[i][0]<=target<=matrix[i][j]:
                s, e = 0, j
                while s <= e:
                    m = s + (e-s) // 2
                    if matrix[i][m] == target:
                        return True
                    elif matrix[i][m] < target:
                        s = m + 1
                    else:
                        e = m - 1
            i -= 1
        return False
```

### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

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

```python
class Solution:
    def findDuplicate(self, nums) -> int:
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow==fast:
                break
        fast = 0
        while True:
            slow = nums[slow]
            fast = nums[fast]
            if slow==fast:
                break
        return slow
```



### [347. 前K个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)（过）

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







## 深度优先遍历

### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)（过）

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



## 栈和队列

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

### [739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

**题目描述：**

根据每日气温列表，请重新生成一个列表，对应位置的输入是你需要再等待多久温度才会升高的天数。如果之后都不会升高，请输入 0 来代替。

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
    def dailyTemperatures(self, T):
        if len(T)==0:
            return []
        res = [0]* len(T)
        stack = []
        for i in range(1,len(T)):
            if T[i]>T[i-1]:
                res[i-1] = 1
                while stack and T[stack[-1]]<T[i]:
                    res[stack[-1]] = i-stack[-1]
                    stack.pop(-1)
            else:
                stack.append(i-1)
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



## 贪心算法

### [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)(过)

**题目描述：**

罗马数字包含以下七种字符： I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内。

```
示例 1:

输入: 3
输出: "III"
示例 2:

输入: 4
输出: "IV"
示例 3:

输入: 9
输出: "IX"
示例 4:

输入: 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
示例 5:

输入: 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        # 使用贪心算法
        roman_list = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
        roman_dic = {1:"I",5:"V",10:"X",50:"L",100:"C",500:"D",1000:"M",
                     4:"IV",9:"IX",40:"XL",90:"XC",400:"CD",900:"CM"}
        i = len(roman_list)-1
        re = ''
        while i>=0:
            if num >= roman_list[i]:   # 找当前最大的
                n = int(num // roman_list[i])   #取最高位数字
                change = n * roman_list[i]  
                if change in roman_dic:
                    re = re + roman_dic[change]
                else:
                    re = re + roman_dic[roman_list[i]]*n
                num = num - n * roman_list[i]
            i -= 1
        return re
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



# 剑指Offer

### 1.青蛙的n级跳

**题目描述：**

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

**解题思路：**

1）这里的f(n) 代表的是n个台阶有一次1,2,...n阶的 跳法数。

2）n = 1时，只有1种跳法，f(1) = 1

3) n = 2时，会有两个跳得方式，一次1阶或者2阶，这回归到了问题（1） ，f(2) = f(2-1) + f(2-2) 

4) n = 3时，会有三种跳得方式，1阶、2阶、3阶，

```
那么就是第一次跳出1阶后面剩下：f(3-1);第一次跳出2阶，剩下f(3-2)；第一次3阶，那么剩下f(3-3)
因此结论是f(3) = f(3-1)+f(3-2)+f(3-3)
```

5) n = n时，会有n中跳的方式，1阶、2阶...n阶，得出结论：

```
f(n) = f(n-1)+f(n-2)+...+f(n-(n-1)) + f(n-n) => f(0) + f(1) + f(2) + f(3) + ... + f(n-1）
```

6) 由以上已经是一种结论，但是为了简单，我们可以继续简化：

```
f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2)
f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1)

可以得出：f(n) = 2*f(n-1)
```

**代码实现**

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number==1:
            return 1
        res = 1
        for i in range(2,number+1):
            res = 2 * res
        return res
```

### 2.2  2*1的小矩形覆盖一个2 *n的大矩形，总共有多少种方法？

题目描述：

我们可以用2  *1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2  *1的小矩形无重叠地覆盖一个2 *n的大矩形，总共有多少种方法？

**解题思路：**

和爬楼梯一样， f(n) = F(n-1)+f(n-2)

### 3.二进制表示中1的个数

**题目描述：**

输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

**解题思路：**

首先判断n是不是负数，当n为负数的时候，直接用后面的while循环会导致死循环，因为负数向左移位的话最高位补1 ！ 因此需要一点点特殊操作，可以将最高位的符号位1变成0，也就是n & 0x7FFFFFFF，这样就把负数转化成正数了，唯一差别就是最高位由1变成0，因为少了一个1，所以count加1。如果是正数那么直接按位与，然后右移一位再继续跟1按位与就可以得到1的个数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        res = 0
        if n<0:
            n = n & 0x7FFFFFFF
            res = res + 1   #这里为了节省时间可以使用 + +count
        while n != 0:
            res += n & 1
            n = n >> 1
        return res
```

### 4.奇数位于数组的前半部分，所有的偶数位于数组的后半部分。

**题目描述：**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

**解题思路：**

使用python内部的filter()方法

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        odd_list = filter(lambda x:x%2!=0, array)
        evev_list = filter(lambda x:x%2==0, array)
        return list(odd_list) + list(evev_list)
```

### 5.输入一个链表，输出该链表中倒数第k个结点。

```python
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        if head is None:
            return head
        dic = {}
        c = 1
        ptr = head
        while ptr:
            dic[c] = ptr
            ptr = ptr.next
            c += 1
        n = len(dic)-k+1
        if n in dic:
            return dic[n]
        else:
            return None
```

### 6.反转链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        pre, ptr = pHead, pHead.next
        pre.next = None    # 注意这句
        while ptr:
            tmp = ptr.next
            ptr.next = pre
            pre = ptr
            ptr = tmp
        return pre
```

### 7.输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if pRoot1 is None or pRoot2 is None:
            return False
        result = False
        if pRoot1.val == pRoot2.val:
            result = self.isSubset(pRoot1, pRoot2)
        if not result:
            result = self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
        return result

    def isSubset(self,root1,root2):
        if root2 is None:
            return True
        if root1 is None:
            return False
        if root1.val==root2.val:
            return self.isSubset(root1.left, root2.left) and self.isSubset(root1.right, root2.right)
        return False
```

### 8.输入某二叉树的前序遍历和中序遍历的结果，请重建二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre)==0 or len(tin)==0:
            return None
        root = TreeNode(pre[0])
        for order, item in enumerate(tin):
            if root.val==item:
                root.left = self.reConstructBinaryTree(pre[1:order+1], tin[:order])
                root.right = self.reConstructBinaryTree(pre[order+1:], tin[order+1:])
                return root
```

**按中序遍历和后序遍历建立二叉树**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, inorder, postorder):
        if len(inorder) == 0 or len(postorder) == 0:
            return None
        root = TreeNode(postorder[-1])
        for index, value in enumerate(inorder):
            if root.val == value:
                root.left = self.buildTree(inorder[:index], postorder[:index])
                root.right = self.buildTree(inorder[index+1:], postorder[index:len(postorder) - 1])
                return root
```



### 9.交换二叉树的左右孩子（源二叉树的镜像）

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if root is None:
            return root
        left = self.Mirror(root.left)
        right = self.Mirror(root.right)
        root.left = right
        root.right = left
        return root
```

### 10.按顺时针遍历二维数组

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        res = []
        if len(matrix)==0 or len(matrix[0])==0:
            return res
        m, n = len(matrix), len(matrix[0])
        j = 0
        while len(res)<m*n:
            res.extend(matrix[j][j:n-j])
            for i in range(j+1, m-j):
                res.append(matrix[i][n-j-1])
            res.extend(matrix[m-j-1][j:n-j-1][::-1])
            for i in range(m-j-2, j, -1):
                res.append(matrix[i][j])
            j +=1
        return res[0:m*n]
```

### 11.栈的压入和弹出

题目描述

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        if len(pushV)==0 or len(popV)==0:
            return False
        stack = []
        for v in pushV:
            stack.append(v)
            while stack and stack[-1]==popV[0]:
                stack.pop()
                popV.pop(0)
        return len(stack)==0
```

### 12.输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。

```python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if len(sequence)==0:
            return False
        index = 0   # 用于记录左右子树的分节点
        for i in range(len(sequence)):
            if sequence[i]>sequence[-1]:
                index = i
                break
        for j in range(i, len(sequence)):
            if sequence[j]<sequence[-1]:
                return False
        left = True
        right = True
        if len(sequence[:index])>0:
            left = self.VerifySquenceOfBST(sequence[:index])
        if len(sequence[index:-1])>0:
            right = self.VerifySquenceOfBST(sequence[index:-1])
        return left and right
```

### 13.Posting List(复杂链表)，有两个指针，next和random

```python
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
class Solution:
    # 返回 RandomListNode
    def __init__(self):
        self.cpHead = None
        self.head = None
    def Clone(self, pHead):
        # write code here
        if pHead is None:
            return pHead
        self.head = pHead
        self.copyNode()
        self.copyRandomNode()
        self.resumList()
        return self.cpHead
	# 第一步复制节点， 改变next指针
    def copyNode(self):    
        ptr = self.head
        while ptr:
            cpNode = RandomListNode(ptr.label)
            cpNode.next = ptr.next
            ptr.next = cpNode
            ptr = cpNode.next
        self.cpHead = self.head.next
	# 复制随机指针
    def copyRandomNode(self):
        ptr = self.head
        while ptr:
            randomNode = ptr.random
            cpNode = ptr.next
            if randomNode:
                cpNode.random = randomNode.next
            ptr = cpNode.next
	# 恢复原来的链表
    def resumList(self):
        ptr = self.head
        while ptr:
            cpNode = ptr.next
            ptr.next = cpNode.next
            if ptr.next:
                cpNode.next = ptr.next.next
            else:
                cpNode.next = None
            ptr = ptr.next
```

### 14.对称的二叉树

```python
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if pRoot==None:
            return True
        result = self.isEqual(pRoot.left, pRoot.right)
        return result

    def isEqual(self, root1, root2):
        if not root1 and not root2:
            return True
        elif root1 and not root2:
            return False
        elif root2 and not root1:
            return False
        if root1.val!=root2.val:
            return False
        left = self.isEqual(root1.left, root2.right)
        right = self.isEqual(root1.right, root2.left)
        return left and right
```

### 15.二叉树中和为某一值得路径

**解题思路：**

我们可以先从最简单的情况开始考虑，最简单的情况就是二叉树只有一个根节点，判断根节点的值与期望值是否相同就ok了。二叉树稍微复杂一点就是根节点还有左右子节点，这时候的过程就要多一步，仍旧是先判断根节点的值与期望值，如果相等或者期望值更小，则不必继续向下判断，如果期望值更大，那么可以向下继续判断，此时的期望值变成了“期望值-根节点的值”，用新的期望值分别与左右子节点的值进行比较，因为左右子节点已经是叶节点，符合路径的定义，因此如果节点的值与新的期望值相等，就得到了答案，如果不相等，问题无解。现在推广到普通的二叉树，与上面的分析相同，就是一个不断更新期望值并与节点值比较的过程，这个过程是重复的，可以利用递归完成，

**代码实现**

```python
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        res = []
        if root is None:
            return res
        def iterPath(root, expectNumber, dic):
            if expectNumber>root.val:
                dic.append(root.val)
                if root.left:
                    iterPath(root.left, expectNumber-root.val, dic)
                if root.right:
                    iterPath(root.right, expectNumber - root.val, dic)
            elif expectNumber==root.val:
                dic.append(root.val)
                if not root.left and not root.right:
                    res.append(dic[:])
            else:
                dic.append(0)
            dic.pop()    # pop操作的目的就是让dic从当前节点重新回到其父亲节点
        iterPath(root, expectNumber, [])
        return res
```

### 16.二叉搜索树和双向链表

按中序遍历二叉树，每遍历一个节点，更新整个双向链表的头指针和尾指针。

```python
class Solution:
    def __init__(self):
        self.Listhead = None
        self.ListTail = None
    def Convert(self, pRootOfTree):
        if pRootOfTree is None:
            return 
        self.Convert(pRootOfTree.left)
        if self.Listhead is None:
            self.Listhead = pRootOfTree
            self.ListTail = pRootOfTree
        else:
            self.ListTail.right = pRootOfTree
            pRootOfTree.left = self.ListTail
            self.ListTail = pRootOfTree
        self.Convert(pRootOfTree.right)
        return self.Listhead
```

### 17.字符串排列

例如：abc, 得到的结果为： abc,acb,bac,bca,cab,cba

```python
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        def solve(ss, soultion):
            if len(ss)==0:
                if soultion not in res:
                    res.append(soultion)
                return
            for i in range(len(ss)):
                newSolution = soultion + ss[i]
                new_ss = ss[0:i]+ss[i+1:]
                solve(new_ss, newSolution)
        res = []
        if len(ss)==0:
            return res
        solve(ss, '')
        return res
```

### 18.整数1中出现的次数

题目描述

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

```python
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        if n<1:
            return 0
        count = 0
        base = 1
        round = n
        while round>0 :
            weight = round%10
            round /= 10
            count += round*base
            if weight==1:
                count+=(n%base)+1
            elif weight>1:
                count += base
            base*=10
        return count
```

### 19.把数组排列成最小的数

题目描述

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```python
# -*- coding:utf-8 -*-
import sys
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if len(numbers)==0:
            return ''
        res = sys.maxsize
        res = self.getMin(numbers,'',res)
        return int(res)

    def getMin(self, numbers, solution, res):
        if len(numbers) == 0:
            if 1 <= int(solution) <= sys.maxsize:
                res = min(res, int(solution))
                return res
            else:
                return res
        for i in range(len(numbers)):
            newSolution = solution + str(numbers[i])
            newNumbers = numbers[0:i]+numbers[i+1:]
            res = self.getMin(newNumbers, newSolution, res)
        return res
```

### 20.统计一个数字在排序数组中出现的次数。

因为data中都是整数，所以可以稍微变一下，不是搜索k的两个位置，而是搜索k-0.5和k+0.5这两个数应该插入的位置，然后相减即可。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        count = 0
        if len(data)==0:
            return count
        def insert(data, num):
            i, j = 0, len(data)-1
            while i<=j:
                m = i + (j - i) // 2
                if data[m] < num:
                    i = m + 1
                elif data[m]>num:
                    j = m - 1
            return i
        i = insert(data, k-0.5)
        j = insert(data, k+0.5)
        return j - i
```

### 21.判断二叉树的平衡性

解题思路

比较每个节点的左子树和右子树的高度差，如果绝对值大于1，则表明该二叉树不平衡

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.balance = True
    def IsBalanced_Solution(self, pRoot):
        # write code here
        self.computeBlance(pRoot)
        return self.balance

    def computeBlance(self,root):
        if root is None:
            return 0
        left = self.computeBlance(root.left)
        right = self.computeBlance(root.right)
        if abs(left - right) > 1:
            self.balance = False
        return max(left, right) + 1
```

### 22.和为S的连续正数序列

题目描述

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        result = []
        plow, phigh = 1, 2
        while plow<phigh:
            cur = int((phigh + plow)*(phigh - plow + 1) / 2)
            if cur == tsum:
                result.append(list(range(plow,phigh+1)))
                plow += 1
            elif cur < tsum:
                phigh += 1
            else:
                plow += 1
        return result
```

### 23.和为S的两个数字

题目描述

输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

```python
# -*- coding:utf-8 -*-
import sys
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code herea
        result = []
        minMultiply = sys.maxsize
        if len(array)==0:
            return result
        i, j = 0, len(array) - 1
        while i < j:
            tmp = array[i] + array[j]
            if tmp == tsum:
                tmpMultiply = array[i] * array[j]
                if tmpMultiply < minMultiply:
                    result = []
                    minMultiply = tmpMultiply
                    result.extend([array[i], array[j]])
                i += 1
                j -= 1
                while i < j and array[i] == array[i - 1]:
                    j += 1
                while i < j and array[j] == array[j + 1]:
                    j -= 1
            elif tmp < tsum:
                i += 1
            else:
                j -= 1
        return result
```

### 24.圆圈中剩下的数

题目描述

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n==0 or m==0:
            return -1
        queue = list(range(0,n))
        while len(queue)>1:
            No = m % len(queue) - 1
            if No==-1:
                queue = queue[0:No]
            else:
                queue = queue[No + 1:] + queue[0:No]
        return queue[0]
```

### 25.求1+2+3+...+4的和

题目描述

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

解析：使用递归完成

```python
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        re = self.getSum(0,n)
        return re
    def getSum(self, result, n):
        if n > 0:
            result += n
            result = self.getSum(result, n - 1)
        return result
```

### 26.不用加减乘除做加法（不懂）

**题目描述**

写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

```python
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        while num2!=0:
            tmp = num1 ^ num2
            num2 = (num1 & num2) << 1
            num1 = tmp & 0xFFFFFFFF
        return num1 if num1 <= 0x7FFFFFFF else ~(num1 ^ 0xFFFFFFFF)
```

### 27.扑克牌顺子

题目描述

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        len_numbers = len(numbers)
        if len_numbers==0:
            return False
        numbers.sort()
        num_0 = [i for i in numbers if i==0]
        tmp_num = numbers[len(num_0):]
        # 判断除0之外的数字有没有重复的
        if len(tmp_num)!=len(set(tmp_num)):
            return False
        # 判断非0数字列表的长度为0时
        if len(tmp_num)==0:
            return True
        # 判断0的列表为空
        elif len(num_0)==0:
            return (tmp_num[-1]-len(tmp_num)+1)==tmp_num[0]
        else:
            # 需要0的数目
            need_num = (tmp_num[-1]-tmp_num[0]+1) - len(tmp_num)
            return need_num <= len(num_0)
```

### 28.把字符串转成整数

题目描述

将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

输入描述:

```
输入一个字符串,包括数字字母符号,可以为空
```

输出描述:

```
如果是合法的数值表达则返回该数字，否则返回0
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        re = 0
        if len(s)==0:
            return re
        flage = 1   # 正数和负数的标志
        if s[0]=='-':
            flage = -1
        s = s.lstrip('+')
        s = s.lstrip('-')
        carry = 1
        i = len(s)-1
        while i>=0:
            if '0'<=s[i]<='9':
                re += carry * int(s[i])
                carry = carry * 10
            else:
                return 0
            i -= 1
        return re * flage
```

### 29.正则表达式匹配

题目描述

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        # 判断匹配规则是否为空
        if pattern == "":
            # p为空的时候，判断s是否为空，则知道返回True 或 False
            return s == ""
        # 判断匹配规则是否只有一个
        if len(pattern) == 1:
            # 判断匹配字符串长度是否为1，和两者的第一个元素是否相同，或匹配规则使用.
            return len(s) == 1 and (s[0] == pattern[0] or pattern[0] == '.')
        # 匹配规则的第二个字符串不为*，当匹配字符串不为空的时候
        # 返回 两者的第一个元素是否相同，或匹配规则使用. and 递归新的字符串(去掉第一个字符的匹配字符串 和 去掉第一个字符的匹配规则)
        if pattern[1] != "*":
            if s == "":
                return False
            return (s[0] == pattern[0] or pattern[0] == '.') and self.match(s[1:], pattern[1:])
        # 当匹配字符串不为空 and (两者的第一个元素是否相同 or 匹配规则使用.)
        while s and (s[0] == pattern[0] or pattern[0] == '.'):
            # 到了while循环，说明p[1]为*，所以递归调用匹配s和p[2:](*号之后的匹配规则)
            # 用于跳出函数，当s循环到和*不匹配的时候，则开始去匹配p[2:]之后的规则
            if self.match(s, pattern[2:]):
                return True
            # 当匹配字符串和匹配规则*都能匹配的时候，去掉第一个字符成为新的匹配字符串，循环
            s = s[1:]
        # 假如第一个字符和匹配规则不匹配，则去判断之后的是否匹配
        return self.match(s, pattern[2:])
```

### 30.表示数值的字符串

```python
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        if s[0]=="+" or s[0]=="-":
            if s[1]!="+" and s[1]!="-":
                try:
                    if float(s[1:]):
                        return True
                except:
                    return False
            else:
                return False
        else:
            try:
                if float(s):
                    return True
            except:
                return False
```

### 31.字符流中第一个不重复的字符

题目描述

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

输出描述:

如果当前字符流没有存在出现一次的字符，返回#字符。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.res = ''
        self.stack = []
        self.first = '#'
    def FirstAppearingOnce(self):
        # write code here
        return self.first

    def Insert(self, char):
        # write code here
        if char not in self.stack:
            self.stack.append(char)
            if self.first=='#':
                self.first = char
        else:
            if char==self.first:
                self.stack.remove(char)
                self.first = self.stack[0] if len(self.stack)>0 else '#'
            else:
                self.stack.remove(char)
```

### 32.删除链表中重复的元素

```python
# -*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead is None or pHead.next is None:
            return pHead
        pre, ptr, dis = pHead, pHead.next, None
        while ptr:
            if pre.val == ptr.val:
                s, e = pre, ptr.next
                while e and e.val ==s.val:
                    e = e.next
                if dis is None:
                    pHead = e
                else:
                    dis.next = e
                ptr = e    # 此时e 可能为空
            else:
                dis = pre
            if ptr:  #判断ptr是否为空
                pre, ptr = ptr, ptr.next
        return pHead
```

### 33.二叉树的下一个节点

解题思路：

```
IF   该节点为单个节点，即没有父节点、左节点、右节点
		return  None
ELIF  该节点为叶子节点：
      IF 该节点为左叶子节点
         return 该节点的父节点
      IF 该节点为右叶子节点
         IF 该节点为左子树的叶节点：
            return 根节点
         IF 该节点为右字数的叶节点：
            return None
Else
     找到右子树的最深的第一个左节点
```



```python
class Solution:
    def GetNext(self, pNode):
        if(pNode.next is None and pNode.left is None and pNode.right is None):
            # 只有一个节点
            return None
        elif pNode.left is None and pNode.right is None:
            # 该节点为叶子节点时
            fatherNode = pNode.next
            if fatherNode.left==pNode:
                # 为左叶节点
                return pNode.next
            if fatherNode.right == pNode:
                # 为右叶节点
                ptr = fatherNode
                while ptr.next:
                    ptr = ptr.next
                return ptr if ptr.right.val > pNode.val else None
        else:
            if pNode.right is None:
                return pNode.next
            else:
                ptr = pNode.right
                while ptr and ptr.left:
                    ptr = ptr.left
                return ptr
```

### 34.序列化二叉树

```python
class Solution:
    def Serialize(self, root):
        # write code here
        res = []
        if root is None:
            return res
        queue = [root]
        while len(queue)>0:
            next_queue = []
            for node in queue:
                if node == '#':
                    res.append('#')
                else:
                    res.append(node.val)
                    if node.left and node.right:
                        next_queue.append(node.left)
                        next_queue.append(node.right)
                    elif node.left and not node.right:
                        next_queue.append(node.left)
                        next_queue.append('#')
                    elif not node.left and node.right:
                        next_queue.append('#')
                        next_queue.append(node.right)
            queue = next_queue[:]
        return res

    def Deserialize(self, s):
        # write code here
        if len(s)==0:
            return None
        if len(s)==1:
            return TreeNode(s[0])
        nodeList = s
        head = TreeNode(nodeList.pop(0))
        rootLayer = [head]

        while len(nodeList) > 0:
            newRootLayer = []
            nodeLayer = nodeList[0:2*len(rootLayer)]
            for root in rootLayer:
                if len(nodeLayer)>0:
                    leftNode = nodeLayer.pop(0)
                    rightNode = nodeLayer.pop(0) if len(nodeLayer)>0 else '#'
                    if leftNode != '#':
                        root.left = TreeNode(int(leftNode))
                        newRootLayer.append(root.left)
                    if rightNode != '#':
                        root.right = TreeNode(int(rightNode))
                        newRootLayer.append(root.right)
            nodeList = nodeList[2 * len(rootLayer):]
            rootLayer = newRootLayer[:]
        return head
```

### 35.在字符串形式的矩阵中查找字符串

题目描述

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if rows==0 or cols==0:
            return False
        dp = []
        while len(matrix)!=0:
            dp.append(list(matrix[0:cols]))
            matrix = matrix[cols:]
        for i in range(rows):
            for j in range(cols):
                if self.helper(dp, i, j, path):
                    return True
        return False

    def helper(self,matrix, i, j, path):
        if len(path)==0:
            return True
        if 0<=i<len(matrix) and 0<=j<len(matrix[0]):
            if matrix[i][j]==path[0]:
                matrix[i][j] = ''
                if self.helper(matrix, i-1, j, path[1:]):
                    return True
                if self.helper(matrix, i+1, j, path[1:]):
                    return True
                if self.helper(matrix, i, j-1, path[1:]):
                    return True
                if self.helper(matrix, i, j+1, path[1:]):
                    return True
                matrix[i][j] = path[0]
        return False
```

### 36.机器人的运动范围

```python
# -*- coding:utf-8 -*-
class Solution:
    def movingCount(self, threshold, rows, cols):
        # write code here
        if rows<1 or cols<1 or threshold <0:
            return 0
        dp = [[0 for _ in range(cols)] for _ in range(rows)]
        count = 0
        #判断第一行
        for j in range(cols):
            if self.every(0) + self.every(j) <= threshold:
                dp[0][j] = 1
                count += 1
            else:
                break
        #判断第一列
        for i in range(rows):
            if self.every(i) + self.every(0) <= threshold:
                dp[i][0] = 1
                count += 1
            else:
                break
        # 判断其他的位置
        for i in range(1, rows):
            for j in range(1, cols):
                if self.every(i) + self.every(j) <= threshold:
                    if dp[i-1][j]!=0 or dp[i][j-1]!=0:  #可到达此位置的条件
                        dp[i][j] = 1
                        count += 1
        return count-1   # 0,0位置多加了一遍

    def every(self,x):  # 计算一个数的各个未的和
        result = 0
        while x!=0:
            result += x % 10
            x  = x // 10
        return result
```







