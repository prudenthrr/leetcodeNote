# python 算法书

### 1.合并两个有序数组

```python
def combine_two_array_method_2(l1, l2):
    res = []
    while l1 and l2:
        if l1[0] < l2[0]:
            res.append(l1.pop(0))
        else:
            res.append(l2.pop(0))
    if l1:
        res.extend(l1)
    else:
        res.extend(l2)

    return res
```

### 2.二分查找

```python
def binarysearch(nums, t):
    b, e = 0, len(nums)-1
    while b<=e:
        m = b + (e - b) // 2   # m = (b + e) // 2
        if nums[m]==t:
            return m
        elif nums[m]<t:
            b = m+1
        else:
            e = m-1
    return -1
nums = [1,3,5,6,7,8,13,14,15,17,18,24,30,43,56]
print(binarysearch(nums, 56))
```

### 3.单词模式匹配

**题目描述：**

例如 pattern = ['a','b','b','a']，str_input = ['1','2','2','1']

a-->> 1    b-->> 2

```python
def word_pattern(pattern,str_input):
    pat_elem = {}
    used_elem = {}

    for i in range(len(pattern)):
        if pattern[i] in pat_elem:
            if pat_elem[pattern[i]] != str_input[i]:
                return False
        elif str_input[i] in used_elem:      #a-->> 1    b-->> 1  该种情况使用另一个字典来标记
            return False
        else:
            pat_elem[pattern[i]] = str_input[i]
            used_elem[str_input[i]] = True
    return True
```

### 4.猜数字

**题目描述：**

secret为正确的数，guess为猜出的数。A  为数字个位置都正确的计数。   B为猜对数字但是位置不对的数字。例如输出结果为1A3B

**解题思路：**

依次比较两个列表元素，如果对应位置的元素相等，则A+=1。 如果不相等则将元素添加到各自所对应的字典中，元素没出现1次加1。最后查看两个字典中的元素，一个字典中的键是否包含在另一个字典中，当键包含时，则取两个字典中键所对应的最小值。

def getHint(secret,guess):
    dic_secret = {}    #正确的数的字典，
    dic_guess = {}     #猜出的数的字典
    A,B = 0,0

```python
for i in range(len(secret)):
    if secret[i] == guess[i]:
        A += 1
    else:
        '''普遍形式'''
        # if secret[i] in dic_secret:
        #     dic_secret[secret[i]] = dic_secret[secret[i]] + 1
        # else:
        #     dic_secret[secret[i]] = 1
        # if guess[i] in dic_guess:
        #     dic_guess[guess[i]] = dic_guess[guess[i]] + 1
        # else:
        #     dic_guess[guess[i]] = 1

        '''简写形式'''
        #初始加入到字典时将值设置为1，再次加入时将原来值增加1
        dic_secret[secret[i]] = dic_secret[secret[i]] + 1 if secret[i] in dic_secret else 1
        dic_guess[guess[i]] = dic_guess[guess[i]] + 1 if guess[i] in dic_guess else 1

for key in dic_secret:
    if key in dic_guess:
        B += min(dic_secret[key], dic_guess[key])   #取键值最小的数

return str(A) + "A" + str(B) + "B"
```

### 5.打家劫舍（别墅抓小偷）

[337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

**题目描述：**

在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

示例 1:

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

示例 2:

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

为每个节点（别墅）设置一个一位数组[偷，不偷]，其中每个节点的

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
        re  = 0
        if root is None:
            return re
        def helper(root):
            if root is None:
                return [0, 0]
            left = helper(root.left)
            right = helper(root.right)
            rob = root.val +left[1]+right[1]
            skip = max(left)+max(right)
            return [rob, skip]
        re = helper(root)
        return max(re)
```

### 6.二叉树中的最大路径

**题目描述：**

给定一个非空二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。

示例 1:

```
输入: [1,2,3]

   1
  / \
 2   3
输出: 6
```

示例 2:

```
输入: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出: 42
```

**解题思路：**

```
（1）每个节点智能访问一次，访问到叶节点时停止。
（2）每个节点有两种可能：继续（yes）和不继续（no）
（3）继续的条件为：线路两端至少至少有一端部位叶节点
（4）yes = max(节点值，节点值+左孩子yes，节点值+右孩子yes)
（5）no = max(左子树yes，右子树yes，左子树no，右子树no，节点值+左子树yes+右子树yes)
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        re = 0
        if root is None:
            return re
        re = self.dfs(root)
        return max(re)
        
    def dfs(self,root):
        import sys
        if root is None:
            return [-sys.maxsize, -sys.maxsize]
        left = self.dfs(root.left)
        right = self.dfs(root.right)
        yes = max(root.val, root.val+left[0], root.val+right[0])
        no = max(left[0], left[1], right[0],right[1], root.val+left[0]+right[0])
        return [yes, no]
```

### 7.寻找最大面积岛屿

[695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

**题目描述：**

给定一个包含了一些 0 和 1的非空二维数组 grid , 一个 岛屿 是由四个方向 (水平或垂直) 的 1 (代表土地) 构成的组合。你可以假设二维矩阵的四个边缘都被水包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为0。)

示例 1:

```
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
对于上面这个给定矩阵应返回 6。注意答案不应该是11，因为岛屿只能包含水平或垂直的四个方向的‘1’。
```

示例 2:

```
[[0,0,0,0,0,0,0,0]]
对于上面这个给定的矩阵, 返回 0。
```

```python
class Solution:
    def maxAreaOfIsland(self, grid) -> int:
        re = 0
        if len(grid)==0 or len(grid[0])==0:
            return 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==1:
                    tmp = self.dfs(grid, i, j, 1)
                    re = max(re, tmp)
        return re

    def dfs(self, grid, i, j, current):
        grid[i][j] = 2   # 修改为2
        if i-1>=0 and grid[i-1][j]==1:
            current = self.dfs(grid, i-1, j, current+1)
        if i+1<len(grid) and grid[i+1][j]==1:
            current = self.dfs(grid, i+1, j, current+1)
        if j-1>=0 and grid[i][j-1]==1:
            current = self.dfs(grid, i, j-1, current+1)
        if j+1<len(grid[0]) and grid[i][j+1]==1:
            current = self.dfs(grid, i, j + 1, current + 1)
        return current
```

### 8.选课系统

**题目描述：**

选课系统，实现依次选修课的列表输出

编号为0,1,2,3,4， 其中只有选修0和1后才可以学习2，选修2和3后才可以选修4

```
# 0---
#      2 ---
# 1---       4
#      3 ---
```

```python
# 广度优先遍历
#选课系统，实现依次选修课的列表输出
#编号为0,1,2,3,4， 其中只有选修0和1后才可以学习2，选修2和3后才可以选修4
# 4---
#      2 ---
# 3---       0
#      1 ---
def bfs(prelist):
    """
    广度优先遍历
    :param prelist:
    :return:
    """
    result = []    #存储结果
    if not prelist:
        return result
    pre_class_count = [0]*len(prelist)      #构建选修课的先先修课数，初始化为0
    for line in prelist:
        for i in range(len(line)):
            if line[i]==1:
                pre_class_count[i] += 1    #填充先修课数列表

    queue = [i for i in range(len(pre_class_count)) if pre_class_count[i]==0]  #将选修课数为0的科目追加到队列

    while queue:
        thisclass = queue.pop(0)    #当前科目代码
        result.append(thisclass)
        for i in range(len(prelist)):
            if prelist[thisclass][i]==1:    #如果当前科目是其他科目的先修课，即 thisclass 是 i 的先修课
                pre_class_count[i] -= 1     #先修课数列表减1
                if pre_class_count[i]==0:  #将先修课数为0的科目追加到队列
                    queue.append(i)
    return result

prelist = [[0,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0]]
print("选修课程代码依次为")
print(bfs(prelist))

#输出结果
# 选修课程代码依次为
# [1, 3, 4, 2, 0]

```

### 11.打印二叉树右侧节点

```python
def print_right_point(root):
    result = []              #存储每层最右边的节点
    if root is None:
        return None
    queue = [root]
    while queue:
        next_queue = []
        len_queue = len(queue)
        for i in range(len_queue):  #循环弹出当前队列
            node  = queue.pop(0)
            if i==len_queue-1:     #取每层的最后一个节点
                result.append(node.val)
            if node.left:          #将左节点加入到下层队列
                next_queue.append(node.left)
            if node.right:
                next_queue.append(node.right)    #将右节点加入到下层队列
        queue = next_queue
        
    return result
```

