# python 面试宝典

## 4.数组和字符串

## 4.1 数组的定位排序

### 4.1.1 题目描述

给定一个数组A和一个下标i，将数组按A[i]重新排列成小与A[i]、等于A[i]和大于A[i]的数组。

### 4.1.2 解题思路

（1）将数组分为两部分：第一部分小于A[i]，第二部分大于等于A[i]

（2）对第二部分再分为两部分：第一部分等于A[i]，第二部分大于A[i]

### 4.1.3 代码实现

```python
def rearrangePivot(array, i):
    if len(array)>1:
        if 0<=i<len(array):     # 判断i的合法性
            tmp = array[i]
            s, e = 0, len(array)-1
            while s < e:
                if array[s] >= tmp:
                    array[s], array[e] = array[e], array[s]
                    e -= 1
                else:
                    s += 1
            s, e = e, len(array)-1
            while s < e:
                if array[s] > tmp:
                    array[s], array[e] = array[e], array[s]
                    e -= 1
                else:
                    s += 1

array = [6,5,5,7,9,4,3,3,4,6,8,4,7,9,2,1]
rearrangePivot(array,5)
print(array)
```

## 4.2 在整型数组中构建元素之后能整数数组长度的子集

### 4.2.1 题目描述

给定一个数组，数组中的元素可能重复，且是乱序排列的。数组的长度为n。设计一个算法，找到一系列下标的集合，使得这些下标所对应的数字的和可以整除数组的长度。即：

​                                                      I = {0,1,2,...,n}

​                                          (A0+A1+...+An) % n==0

### 4.2.2 解题思路

将该题想象成一个在0到n-1个盒子里装小球。

![1560340574493](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560340574493.png)

### 4.2.3 代码实现

```python
def findModuleSubset(A):
    if len(A)==0:
        return A
    res = []   # 存放满足条件子集
    box = {}    # 存放余数的字典，键为0到len(A)-1，值同一个余数的下标所构成的列表
    len_A = len(A)
    subset = []   # 存放下标
    sum = 0   # 当前的和
    for i in range(len(A)):
        sum += A[i]
        t = sum % len_A   # 求得的余数
        subset.append(i)
        if t == 0:
            res.append(subset[:])
            if t not in box:
                box[t] = [i]
            else:
                for j in box[t]:  # 遍历当前余数所对用的下标
                    if j !=0 :
                        res.append(subset[j+1:])
                box[t] = box[t] + [i]   # 追加当前下标到余数所对应的列表
        elif t not in box:
            box[t] = [i]
        else:
            for j in box[t]:
                re.append(subset[j+1:])
            box[t] += [i]
    return res

import random
A = [153, 940, 852, 632, 237, 516, 126, 546, 838, 757]
print(findModuleSubset(A))
```

## 4.3 计算等价类

### 4.3.1 题目描述

假设有集合S：{1,2,3...,n-1}，同时有两个数组A，B，它们的元素都来自于集合S，而且长度都为m。A，B两个数组用来确定集合S中的等价类，假如A[k]和B[k]是等价的，那么S就会划分为几个不相交的等价类。

### 4.3.2 解题思路

使用变量S来存储原始0到n-1的元素集合。循环0到m-1，如果处于当前位置的两个元素在已存在的等价类集合中可以找到，则将其归并到对应的等价类中；如果找不到，则为新的等价集合类。

### 4.3.3 代码实现

```python
def makset(A,B,n,m):
    '''
    :param A: 第一个数组
    :param B: 第二个数组
    :param n: 原来集合的元素为0到n-1
    :param m: A和B数组的长度
    :return:
    '''
    S = set()   # 生成原始集合
    res = []    # 存放等价子类  
    for i in range(n):
        S.add(i)
    for i in range(m):
        flag = False    # 判断当前的元素是否是已存在的等价类中的元素
        a, b = A[i], B[i]
        #遍历已存在的等价类
        for subset in res:  
            # 如果当前元素可在等价类中找到
            if a in subset or b in subset:
                subset.add(a)
                subset.add(b)
                flag = True
                break
        if flag==False:  
            res.append({a,b})
    for i in res:
        S = S - i
    for s in S:
        res.append({s})
    return res

A = [1,5,3,6]
B = [2,1,0,5]
m, n = 4, 10
res = makset(A,B,n,m)
print('所有的等价类：', res)
# 所有的等价类： [{1, 2, 5, 6}, {0, 3}, {8}, {9}, {4}, {7}]
```

## 4.4 大型整数相加、相乘

### 4.4.1 字符串数字相加

#### （1）题目描述

给定两个非空字符串数字，不要将字符串转成int类型进行操作，函数返回也是字符串。

#### （2）解题思路

逆序遍历每个字符串，将每位的字符转成int类型来做。使用变量carry来存储进位数。

#### （3）代码实现

```python
def stringAdd(num1, num2):
    # 判断有0的情况
    if num1=='0' and num2=='0':
        return '0'
    elif num1=='0' and num2!='0':
        return num2
    elif num1!='0' and num2=='0':
        return num1
    i, j = len(num1) - 1, len(num2) - 1
    carry = 0
    res = ''
    while i>=0 and j>=0:
        s = carry + int(num1[i]) + int(num2[j])
        carry = s // 10
        res = str(s % 10) + res
        i -= 1
        j -= 1
    while i>=0:
        s = carry + int(num1[i])
        carry = s // 10
        res = str(s % 10) + res
        i -= 1
    while j >= 0:
        s = carry + int(num2[j])
        carry = s // 10
        res = str(s % 10) + res
        j -= 1
    if carry > 0:
        res = str(carry) + res
    return res

num1 = '1234'
num2 = '5678'
print(stringAdd(num1, num2))
```

### 4.4.2 字符串数字相乘

#### （1）题目描述

实现两个数字字符串的乘积，不能直接转换成整数处理。如果数字超过32位，直接转换成数字会出错。

#### （2）解题思路

首先将第一个数字字符串的每个字符（逆序）与第二个数字字符串的最后一个字符相乘，将每个相乘后产生的值进行字符串合并，并在末尾追加0个**‘0’**；然后将第一个数字字符串的每个字符（逆序）与第二个数字字符串的倒数第二个字符相乘，将每个相乘后产生的值进行字符串合并，并在末尾追加1个**‘0’**；依次类推

#### （3）代码实现

```python
class StringMultiply():
    def __init__(self, x , y):
        self.x = x
        self.y = y

    # 乘法运算
    def doMultiply(self):
        if len(self.x)==0 or len(self.y)==0:
            return '0'
        elif self.x == '0' or self.y=='0':
            return '0'
        add_res = []  # 存放与第二个字符串相乘后所得的结果
        len_y, len_x = len(self.y), len(self.x)
        for i in range(len_y-1, -1, -1):    #逆序遍历y
            tmp_sum = ''  # 一次相乘的结果
            carry = 0
            for j in range(len_x-1, -1, -1):  # 逆序遍历x
                tmp_multiply = int(self.x[j]) * int(self.y[i]) + carry
                carry = tmp_multiply // 10
                tmp_sum = str(tmp_multiply % 10) + tmp_sum
            tmp_sum += '0'*(len_y-1-i)
            add_res.append(tmp_sum)
        return self.doListAdd(add_res)

    # 做加法运算
    def doListAdd(self, add_res):
        a = add_res[-1]
        i = len(add_res)-2
        while i>=0:
            b = add_res[i]
            # 做两个字符串的加法运算
            a = self.stringAdd(a,b)
            i -= 1
        return a

    # 做两个字符串的加法运算
    def stringAdd(self, x, y):
        i, j = len(x)-1, len(y)-1
        carry = 0
        sum = ''
        while i>=0 and j>=0:
            add = int(x[i]) + int(y[j]) + carry
            carry = add // 10
            sum = str(add % 10) + sum
            i -= 1
            j -= 1
        # y已经遍历完
        while i>=0:
            add = int(x[i]) + carry
            carry = add // 10
            sum = str(add % 10) + sum
            i -= 1
        # x 已经遍历完
        while j >= 0:
            add = int(y[j]) + carry
            carry = add // 10
            sum = str(add % 10) + sum
            j -= 1
        # x，y都遍历完后，carry的值不为0
        if carry>0:
            sum = str(carry) + sum
        return sum

x, y = '1234', '5678'
print(StringMultiply(x,y).doMultiply())
# 7006652
```

## 4.5数组的序列变换

### 4.5.1 题目描述

给定数组 A = [1,2,3,4,5,6]，P = [3,1,5,4,0,2]，P中的元素表示A中的下标。将数组A根据P的下标重新排序为A=[4, 2, 6, 5, 1, 3]

### 4.5.2 算法实现的关键

（1）如何将被覆盖的元素寄存在数组中

（2）如何找到被移动的元素

### 4.5.3 解题思路

**定理：**对于**P[i]**，如果**P[0.....i-1]**中比**P[i]**大的元素有**k**个，则A[i]元素向右移动了k位。即change = p[i]+k

（1）使用一个变量change来存储当前覆盖元素A[i]的下标，使用tmp变量来存储要覆盖的当前元素。

change = p[i]+原来的A[i]移动的位数， tmp = A[change]

（2）将i到change-1的元素向后移动一位

（3）将tmp的值赋值给原始数组的当前的i位置

### 4.5.4 代码实现

```python
class ArrayPermutation:
    def __init__(self,A, P):
        self.A = A
        self.P = P

    # 重新排序
    def doPermutation(self):
        for i in range(len(self.P)):
            change = self.relocate(i)
            tmp = self.A[change]
            # self.makeshift(i,change)
            # 使用数组的切片操作实现
            self.A[i+1:change+1] = self.A[i:change]  
            self.A[i] = tmp
        return self.A

    # 获得change
    def relocate(self,i):
        '''
        检测在P[0--i-1]中比P[i]大的元素的个数
        :param i:
        :return: int
        '''
        change = self.P[i]
        c = 0
        j = 0
        while j<i:
            if self.P[j] > self.P[i]:
                c += 1
            j += 1
        return change + c

    # 保留被覆盖的元素
    def makeshift(self, begin, end):
        '''
        对i到change-1的元素向后移动一位
        :param begin:
        :param end:
        :return:
        '''
        i = end
        while i>begin:
            self.A[i] = self.A[i-1]
            i -= 1

A = [1,2,3,4,5,6]
P = [3,1,5,4,0,2]
s = ArrayPermutation(A, P)
print(s.doPermutation())
```

## 4.6 字符串的旋转

### （1）给定字符串S和旋转位置i

```python
def rotatestring(s, i):
    if len(s)==0:
        return s
    #判断i的合法性
    if i<0 or i>=len(s):
        return None
    if i==0:
        return s
    s_list = list(s)
    move_len = len(s)-i
    tmp = s_list[i:len(s)]
    s_list[move_len:len(s)] = s_list[0:i]
    s_list[0:move_len] = tmp
    return ''.join(s_list)

s = 'ab'
print(rotatestring(s,1))
```

### （2）给定字符串S、字符串的长度len，旋转位置i

```python
def rotatestring(s, len_s, i):
    if len_s==0:
        return s
    #判断i的合法性
    if i<0 or i>=len_s:
        return None
    if i==0:
        return s
    s = s + s
    return s[i:len_s+i]

s = 'abcdefgh'
print(rotatestring(s,len(s),4))
```

## 4.7 解数独

### 4.7.1 题目描述

leetcode 37

解数独。同一行，同一列，同一个3*3的九宫格内只有1到9的数字，且不重复。

![img](http://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Sudoku-by-L2G-20050714.svg/250px-Sudoku-by-L2G-20050714.svg.png)

### 4.7.2 解题思路

使用回溯法。

### 4.7.3 代码实现

```python
class Solution():
    def solveSudoku(self, board):
        if len(board)==0 or len(board[0]==0):
            return
        self.solve(board)

    # 解决方案
    def solve(self,board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]=='.':
                    for v in range(1, 10):
                        if self.isvalid(board, i, j, v):
                            board[i][j] = v
                            if self.solve(board):
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True
                        
    # 验证所填数字
    def isvalid(self, board, i, j, v):
        # 判断行
        for c in range(len(board[0])):
            if board[i][c]==v:
                return False
        # 判断列
        for r in range(len(board)):
            if board[r][j]==v:
                return False
        # 判断九宫格
        for r in range(i-i%3, i-i%3 + 3):
            for c in range(j-j%3, j-j%3 + 3):
                if board[r][c]==v:
                    return False
        return True

```

## 4.8 二维数组的螺旋遍历

对应的leetcode[54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

### 4.8.1 题目描述

给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。

示例 1:

```
输入:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]

```

示例 2:

```
输入:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
输出: [1,2,3,4,8,12,11,10,9,5,6,7]

```

### 4.8.2 解题思路

如果可以使用第三方工具包numpy，可直接使用矩阵的切片操作来完成。如果不可以使用numpy，则从上右下左的顺序添加元素，但是要注意的是在添加下边的元素和左边的元素时，要逆序添加。以绕圈（j）的形式进行添加元素，j=0, j=1, ...

### 4.8.3 代码实现

```python
# 使用numpy
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        import numpy as np
        res = []
        if len(matrix)==0:
            return res
        matrix = np.array(matrix)
        m, n, j = len(matrix), len(matrix[0]), 0
        while len(res)<m*n:
            res.extend(matrix[j, j:n - j])
            res.extend(matrix[j + 1:m - j, n - j - 1])
            res.extend(matrix[m - j - 1, j:n - j - 1][::-1])
            res.extend(matrix[j + 1:m - j - 1, j][::-1])
            j+=1
        return res[0:m*n]

matrix = [[1,2,3],[5,6,7],[9,10,11]]
print(Solution().spiralOrder(matrix))

# 不使用numpy
class Solution():
    def spiralOrder(self, matrix):
        if len(matrix)==0 or len(matrix[0])==0:
            return matrix
        res = []
        m, n = len(matrix), len(matrix[0])
        j = 0
        while len(res)<m*n:
            res.extend(matrix[j][j:n-j])
            for r in range(j+1, m-j):
                res.append(matrix[r][n-j-1])
            res.extend(matrix[m-j-1][j:n-j-1][::-1])
            for r in range(m-j-2, j, -1):
                res.append(matrix[r][j])
            j += 1
        return res[0:m*n]

```

## 4.9矩阵的90度旋转

leetcode链接[48. 旋转图像](https://leetcode-cn.com/problems/rotate-image/)

### 4.9.1 题目描述

给定一个 n × n 的二维矩阵表示一个图像。

将图像顺时针旋转 90 度。

说明：

你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。

示例 1:

```
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

```

示例 2:

```
给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]

```

### 4.9.2 解题思路

观察最后的结果矩阵发现，第一行的元素为原来矩阵的【第三行的第一个元素，第二行的第一个元素，第一行的第一个元素】 ，依次类推。。。可以使用python的内部函数zip来聚合每个列表的元素，然后将形成的新列表进行翻转。

### 4.9.3 代码实现

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        # 使用map和zip函数
        '''
        matrix[::-1] = [[7,8,9],[4,5,6],[1,2,3]]
        zip函数，合并功能的函数.zip()配合*号操作符,可以将已经zip过的列表对象解压
        map函数，映射功能的函数，将每个迭代对象的每个元素应用到function上，在此处的function为list
        '''
        matrix[:] = map(list,zip(*matrix[::-1]))

```

## 4.10游程编码

### 4.10.1 题目描述

RLE(游程编码)是一种在线高效的压缩算法。例如：’aaaabcccaa‘压缩后变为’4a1b3c2a‘

### 4.10.2 解题思路

用一个变量c来存放相同连续字符出现的次数。在对字符串遍历完之后，结果追加最后一个连续相同的字符。

### 4.10.3 代码实现

```python
def RLEcode(s):
    if len(s)==0:
        return s
    res = ''
    c = 0
    for i in range(len(s)):
        if i!=0 and s[i]!=s[i-1]:
            res = res + str(c)+s[i-1]
            c = 0
        c += 1
    return res+str(c)+s[i]

print(RLEcode('aaaabcccaa'))
#4a1b3c2a

```

## 4.11 字符串中单词的逆转

### 4.11.1题目描述

给定一个字符串，它由若干单词组成，每个单词以空格分开，对对该字符串中的单词进行逆转。例如字符串‘Alice like Bob’，逆转之后就变成‘Bob like Alice’。要求空间复杂度为O(1)

### 4.11.2解题思路

第一步：先将字符串逆转；第二步：对字符串中的单词进行逆转。

### 4.11.3 代码实现

```python
def rotateWord(s):
    if len(s)==0:
        return s
    s = s[::-1]
    res = []
    tmp_str = ''
    for i in range(len(s)):
        if s[i]!=' ':
            tmp_str += s[i]
        else:
           res.append(tmp_str[::-1])
           tmp_str = ''
    res.append(tmp_str[::-1])
    return ' '.join(res)

print(rotateWord('Alice like Bob'))
# Bob like Alice

```

第5章   队列和链表

## 5.1 链表快速倒转

### 5.1.1 题目描述

对链表  1->2->3->4->5->null翻转，翻转后的结果变为5->4->3->2->1

### 5.1.2 解题思路

使用三个指针来完成。使用pre指针来记录当前节点的前一个元素， 使用ptr记录当前节点，使用tmp指针来记录当前节点的后一个节点。然后将当前节点的下一个节点指向前一个节点，然后从tmp节点开始下一次旋转。

### 5.1.3 代码实现

迭代实现

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def rotateList(head):
    if head==None or head.next==None:
        return head
    pre, ptr = head, head.next
    pre.next = None
    while ptr:
        tmp = ptr.next
        ptr.next = pre
        pre = ptr
        ptr = tmp
    return pre

head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
print(rotateList(head))
```

递归实现

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def rotateList(head):
    if head==None or head.next==None:
        return head
    pre = solve(head, head.next)
    return pre

def solve(pre, ptr):
    if ptr==None:
        return pre
    tmp = ptr.next
    ptr.next = pre
    pre = ptr
    pre = solve(pre, tmp)
    return pre

head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
print(rotateList(head))
```

## 5.2 链表成环检测

### 5.2.1 题目描述

检测给定链表是否形成一个环。如果是，**给出构成循环节点个数**。要求算法时间复杂度是O(n)，空间复杂度为O(1).

### 5.2.2 解题思路

![1560657263206](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560657263206.png)

![1560657364070](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560657364070.png)

### 5.2.3 代码实现

输出环的节点个数

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

#  重点  检测循环链表
def examineCircle(head):
    h1, h2 = head, head   # h1一次走一个节点，h2一次走两个节点
    stepCount, visiteCount = 0, 0   # stepCount为一共走的次数， visiteCount两个指针相遇的次数
    first, second = 0, 0   # first为两指针第一次相遇所走的次数， second为第二次相遇所走的次数
    while visiteCount<2:
        if h1==None or h1.next==None or h2==None \
                or h2.next==None or h2.next.next==None:
            return 0
        h1 = h1.next
        h2 = h2.next.next
        stepCount += 1
        if h1==h2:
            visiteCount += 1
            if visiteCount==1:
                first = stepCount
            if visiteCount==2:
                second = stepCount
    return second - first

# 创建循环链表
def creatList(nodeNum):
    if nodeNum<=0:
        return None
    head, val, node = None, 0, None
    tail = Node
    while nodeNum>0:
        if head==None:
            head = Node(val)
            node = head
        else:
            node.next = Node(val)
            node = node.next
            if val==4:
                tail = node
        val += 1
        nodeNum -= 1
    node.next = tail   # 构成循环链表的起点
    return head

head = creatList(9)
print(examineCircle(head))   # 5
```

检测是否成环的代码如下

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return False
        h1, h2 = head, head
        while True:
            if h1 is None or h1.next is None or h2 is None or h2.next is None or h2.next.next is None:
                return False
            h1 = h1.next
            h2 = h2.next.next
            if h1==h2:
                return True
```

## 5.3 在O(1)时间内删除单链表非末尾节点

### 5.3.1 题目描述

给定一个单向链表中的非末尾的节点，要求删除该节点。

### 5.3.2  解决思路

将要删除节点的值改为下一个节点的值，删除下一个节点即可。

### 5.3.3 代码实现

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def deleteNode(node):
    if node.next==None:
        return
    node.val = node.next.val
    node.next = node.next.next
```

## 5.4 获取重合列表的第一个相交节点

### 5.4.1 题目描述

![1560672866175](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560672866175.png)

设计一个时间复杂度为O(n)， 空间复杂度为O(1)的算法，返回两个链表相交时第一个节点，对于上图来说就是返回值为4的节点。

### 5.4.2 解题思路

方法2参考评论。通过巧妙的方法，使两个链表到达相等位置时走过的是相同的距离。时间复杂度还行，空间复杂度不知道为啥又很一般

![å¾®ä¿¡å¾ç_20190531161836.jpg](https://pic.leetcode-cn.com/3aa8a5100e239cf1f63f2990b24d2eabbc8c40c58cc8a57e8c33a214d92d3022-%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20190531161836.jpg)

链表1的长度是x1+y，链表2的长度是x2+y，我们同时遍历链表1和链表2，到达末尾时，再指向另一个链表。则当两链表走到相等的位置时：**x1+y+x2 = x2+y+x1**

没交点则y=0, 结尾都指向None。

### 5.4.3 代码实现

```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA==None or headA==None:
            return 
        p, q = headA, headB
        while p!=q:
            p = p.next if p else headB
            q = q.next if q else headA
        return p
```

## 5.5 单向列表的奇偶排序

### 5.5.1 题目描述

如下图所示，该链表是一个含有奇数和偶数的链表。

![1560675653656](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560675653656.png)

设计一个算法将奇数的节点排在最后，偶数的节点排在前，如下图所示

![1560675709149](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560675709149.png)

### 5.5.2 解题思路

使用e_b, e_e来存储偶数块的起始节点和末尾节点，使用o_b, o_e来存储奇数块的起始节点和末尾节点。循环链表中的每个节点，当该节点的值为偶数时，将该节点加到偶数块末尾节点的后面，更新偶数块的起始节点和末尾节点；当该节点的值为奇数时，记录奇数快的起始节点和末尾节点，然后读取下一个节点。这样，整个链表最后的格局就变为所有的偶数在前面，奇数在后面。

### 5.5.3 代码实现

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
def oddEvenSort(head):
    if head is None or head.next is None:
        return
    e_b, e_e = None, None  # 偶数的开始节点，结尾节点
    o_b, o_e = None, None   # 奇数快的开始节点， 结尾节点
    ptr= head
    while ptr:
        # 操作当前值为偶数的节点
        if ptr.val % 2==0:
            # 偶数块的开始节点为空
            if e_b is None :
                e_b, e_e = ptr, ptr
                if o_b is not None and o_e is not None :
                    # 将偶数块的起始节点变为整个链表的头结点
                    o_e.next, e_b.next,  ptr = ptr.next, o_b, o_e
            else:
                # 奇数快的开始节点不存在
                if o_b is None:
                    e_e = ptr
                else:
                    o_e.next, ptr.next, e_e.next = ptr.next, o_b, ptr
                    e_e, ptr = ptr, o_e
        else:
            if o_b is None:
                o_b, o_e = ptr, ptr
            else:
                o_e = ptr
        ptr = ptr.next
    return e_b if e_b else head

def initList(nums):
    head = Node(nums[0])
    ptr = head
    for i in range(1, len(nums)):
        ptr.next = Node(nums[i])
        ptr = ptr.next
    return head

def printList(head):
    ptr = head
    res = []
    while ptr:
        res.append(ptr.val)
        ptr = ptr.next
    return res

nums = [0,1,2,3,4,5,6,7,8,9]
head = initList(nums)
print('原来链表：',printList(head))
new_head = oddEvenSort(head)
print('重新排序后的链表：',printList(new_head))
```

## 5.6 双指针单向链表（Posting List）的自我复制

### 5.6.1 题目描述

下图就是一个Posting List。

![1560735114329](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560735114329.png)

要求设计一个算法，复制给定的Posting List。 算法的时间复杂度为O(n)，算法除了分配节点所需的内存之外， 不能分配多余内存。算法可以更改原来队列，但更改后需要将队列恢复原状。

### 5.6.2 解题思路

如果采用先复制节点和next指针，然后再复制jump指针，时间复杂度为O(n2)。如果时间复杂度要求为O(n)，则需要采用以下步骤：

第一、复制节点。将原节点的next指针指向复制的节点，复制节点的next指针指向原节点的下一个节点。

第二、修改复制节点的jump指针。遍历原节点，将复制节点的jump指针指向原节点的jump节点的复制节点。

第三、恢复原来链表。将原节点的next指向复制节点的next节点，将复制节点的next指针指向原节点的next节点的复制节点。

### 5.6.3 代码实现

复制原来链表的代码

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

初始化链表和打印链表

```python
class ListUtility():
    def __init__(self):
        self.head = None
        self.tail = None
        self.node_dic = {}

    def initList(self, nums):
        self.head = Node(nums[0])
        ptr = self.head
        self.node_dic[nums[0]] = ptr
        for i in range(1, len(nums)):
            ptr.next = Node(nums[i])
            ptr = ptr.next
            self.node_dic[nums[i]] = ptr
        return self.head

    def initJumpNode(self):
        ptr = self.head
        ptr.jump = self.node_dic[4]
        ptr = ptr.next
        ptr.jump = self.node_dic[1]
        ptr = ptr.next
        ptr.jump = self.node_dic[0]
        ptr = ptr.next
        ptr.jump = self.node_dic[0]
        ptr = ptr.next
        ptr.jump = self.node_dic[3]
        
 def printList(head):
    ptr = head
    while ptr:
        if ptr.jump is None:
            print('(node val:{0} jump val:null)'.format(ptr.val))
        else:
            print('(node val:{0} jump val:{1})'.format(ptr.val, ptr.jump.val))
        ptr = ptr.next

```

打印结果

```python
'''
原来链表为：
(node val:0 jump val:4)
(node val:1 jump val:1)
(node val:2 jump val:0)
(node val:3 jump val:0)
(node val:4 jump val:3)
恢复原来的链表：
(node val:0 jump val:4)
(node val:1 jump val:1)
(node val:2 jump val:0)
(node val:3 jump val:0)
(node val:4 jump val:3)
复制后的链表：
(node val:0 jump val:4)
(node val:1 jump val:1)
(node val:2 jump val:0)
(node val:3 jump val:0)
(node val:4 jump val:3)
'''

```

## 5.7 按层打印二叉树

### 5.7.1 题目描述

![1560740899648](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560740899648.png)

如图所示是一个二叉树，按层遍历的方式来打印二叉树，例如上图打印的结果为：5,3,7,1,4,6,8,0,2,9

### 5.7.2 解题思路

使用队列来解决该问题。将每层的节点加到对队列中，没次只打印队列中的节点的值。

### 5.7.3 代码实现

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def printBinaryTree(root):
    if root==None:
        return 
    queue = [root]
    res = []
    while len(queue)>0:
        next_queue = []
        for node in queue:
            res.append(node.val)
            if node.left:
                next_queue.append(node.left)
            if node.right:
                next_queue.append(node.right)
        queue = next_queue[:]
    return res

```

## 第 6 章  堆栈和队列

## 6.1 利用堆栈计算逆向波兰表达式

### 6.1 题目描述

给定一个逆向波兰表达式，例如"3,4,*,1,2,+,+"，求出结果

### 6.1.2 解题思路

使用字典来存储操作符（+，-，*，/），使用lambda函数来对相应的操作符进行运算。使用stack来存储非操作运算符。将字符串以’，‘ 的形式分割。然后遍历分割结果。当遇到运算符时，在栈中弹出两个数字来完成运算；如果遇到的是数字则压入栈。

### 6.1.3 代码实现

```python
import operator
def getResult(expression):
    if len(expression)==0:
        return
    exprs = expression.split(',')
    stack = []
    dic = {'*':(lambda x,y:x*y), '+':(lambda x,y:x+y), '-': (lambda x,y:x-y),
           '/':(lambda x,y:x/y)}
    # dic = {'*':operator.mul, '+':operator.add, '-': operator.sub, '/':operator.truediv}
    for i in exprs:
        if i not in dic:
            stack.append(i)
        else:
            y = eval(stack.pop())
            x = eval(stack.pop())
            res = dic[i](y, x)
            stack.append(str(res))
    return stack[-1]

expression = "3,4,*,1,2,+,+"
print(getResult(expression))
```

## 6.2 计算堆栈当前元素最大值

### 6.2.1 题目描述

给定一个堆栈，请给出时间复杂度为O(1)的max实现。

### 6.2.2 解决思路

（1）每次压入栈堆栈时，用一个变量记录当前最大值

（2）使用一个新的堆栈maxstack，当压入元素的值大于当前值时，就将该元素压入栈。

（3）弹出一个元素时，如果弹出的元素是当前最大值，那么把maxstack顶部的元素也弹出。

### 6.2.3 代码实现

```python
import sys

class MaxStack:
    def __init__(self):
        self.stack = []
        self.maxStack = []
        self.maxVal = -sys.maxsize-1

    def push(self, value):
        self.stack.append(value)
        if value > self.maxVal:
            self.maxVal = value
            self.maxStack.append(value)

    def peek(self):
        return self.stack[-1]

    def pop(self):
        if self.peek()==self.maxVal:
            self.maxStack.pop()
            self.maxVal = self.maxStack[-1]
        return self.stack.pop()

    def getMax(self):
        return self.maxVal

ms = MaxStack()
ms.push(5)
ms.push(4)
ms.push(2)
ms.push(3)
print('当前最大值：', ms.getMax())
ms.push(6)
ms.push(1)
ms.push(10)
ms.push(8)
print('当前最大值：', ms.getMax())
ms.pop()
print('当前最大值：', ms.getMax())
ms.pop()
print('当前最大值：', ms.getMax())
'''
当前最大值： 5
当前最大值： 10
当前最大值： 10
当前最大值： 6
'''
```

## 6.3 使用堆栈判断括号匹配

### 6.3.1 题目描述

给定一个括号字符串’((())(()))‘，判断左右括号是否匹配

### 6.3.2 解决思路

当遇到**’(‘**时，将该字符压入栈，当遇到**’)‘**时将栈中的左括号弹出栈，如果此时栈为空，则括号不匹配。如果当字符串遍历完后，则堆栈不为空， 则括号不匹配。

### 6.3.3 代码实现

```python
def ismatch(s):
    if len(s)==0:
        return
    stack = []
    for i in s:
        if i=='(':
            stack.append(i)
        else:
            if len(stack)==0:
                return False
            else:
                stack.pop()
    if len(stack)!=0:
        return False
    return True

s = '((())(()))'
print(ismatch(s))
```

## 6.5 堆栈元素的在线排序

### 6.5.1 题目描述

栈的操作有pop，push，peek，empty。要求只能使用这几种堆栈操作 ，在不分配新的内存的情况下，将栈中的元素从大到小排序。

### 6.5.2 解决思路

通过递归调用的方法，将弹出的元素暂时保存在调用堆栈上。使用Insert方法，每次讲一个元素插入到栈中。

### 6.5.3 代码实现

```python
class StackSort:
    def sort(self, stack):
        if len(stack)==0:
            return stack
        v = stack.pop()
        stack = self.sort(stack)
        stack = self.insert(stack, v)
        return stack

    def insert(self, stack, val):
        if len(stack)==0 or val<=stack[-1]:
            stack.append(val)
            return stack
        v = stack.pop()
        stack = self.insert(stack, val)
        stack.append(v)
        return stack

stack = [1,2,3,4,5,6]
print(StackSort().sort(stack))
```

## 6.6 滑动窗口的最大值

### 6.6.1 题目描述

给定一个数组和一个滑动窗口的大小，请找出所有滑动窗口里的最大值。例如，如果输入数组 {2，3，4，6，2，5，1} 及滑动窗口的大小 3，那么一定存在 6 个滑动窗口，它们的最大值分别为 {4，4，6，6，6，5}。

### 6.6.2 解题思路

使用size来表示滑动窗口的大小。在给定的数组在，从size位置开始遍历数组，每次去size范围内的最大值。

### 6.6.3 代码实现

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        res = []
        if len(num)==0 or size<1 or size>len(num):
            return res
        if size==1:
            return num
        i = size
        while i<=len(num):
            res.append(max(num[i-size:i]))
            i += 1
        return res
print(Solution().maxInWindows([2,3,4,2,6,2,5,1],3))
```

## 6.7 使用堆栈模拟队列

### 6.7.1 题目描述

使用两个堆栈模拟队列时，必须支持两种操作：enqueue和dequeue。前者在队列末尾追加一个元素，后者在队列头部弹出一个元素。要求空间复杂度为O(1)，并且进行m次enqueue和dequeue操作时，时间复杂度必须为O(m)。

### 6.7.2 解题思路

使用一个堆栈A来压入数据，使用另一个堆栈B来压入栈A弹出的元素。

### 6.7.3  代码实现

```python
class StackQueue:
    def __init__(self):
        self.A = []
        self.B = []

    def enqueue(self,val):
        self.A.append(val)

    def dequque(self):
        if len(self.B)==0:
            for i in range(len(self.A)):
                self.B.append(self.A.pop())
        return self.B.pop()

sq =  StackQueue()
print('进入队列：')
for i in range(6):
    sq.enqueue(i)
    print('{0}'.format(i), end='  ')
print('\n出队列：')
for i in range(3):
    print(sq.dequque(), end='  ')
'''
进入队列：
0  1  2  3  4  5  
出队列：
0  1  2 
'''
```

## 9 查找法

## 9.1二分查找法正确代码实现

主要改变是将（b+e） // 2 变为 b + (e-b)//2

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
print(binarysearch(nums, 13))
```

## 9.2 在lg(k)时间内查找两个排序数组合并后第k小元素

该题目前没有做出来。

## 9.3 以下代码实现了在O(k)时间内找到第k小元素

```python
def solve(nums1,  nums2, k):
    if len(nums1)==0 and len(nums2)==0:
        return -1
    elif len(nums1)==0 and len(nums2)!=0:
        if 0<=k-1<len(nums2):
            return nums2[k-1]
        else:
            return -1
    elif len(nums2)==0 and len(nums1)!=0:
        if 0<=k-1<len(nums1):
            return nums1[k-1]
        else:
            return -1
    elif k<=0 or k>len(nums1)+len(nums2):
        return -1
    i, j = 0, 0
    c = 0  # 记录k
    while i<len(nums1) and j<len(nums2):
        if nums1[i]<nums2[j]:
            c += 1
            if c == k:
                return nums1[i]
            i += 1
        else:
            c += 1
            if c == k:
                return nums2[j]
            j += 1
    while i<len(nums1):
        c += 1
        if c == k:
            return nums1[i]
        i += 1
    while j<len(nums2):
        c += 1
        if c == k:
            return nums2[j]
        j += 1

nums1 = [1,3,5,7,9]
nums2 = [2,4,6,8,10]
print(solve(nums1, nums2, 6))  # 6
```

## 9.4 使用二分查找法寻求数组的截断点

### 9.4.1 题目描述

假定有5位员工的薪资分别是90，30,100,40,20， T设置为210，那么截断值可以设定为60，于是高于60 的值变为60，低于60 的值保持不变，由此员工收入变为60，30,60,40,20，贾总正好是210。**要求设计一个算法来找到截断值。**

### 9.4.2 解题思路

![1560607655045](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560607655045.png)

使用二分查找法来解决该题。先对数组进行排序，在使用一个额外数组salariesSum来存储i位置以前以及i位置的元素和。totalCap总工资，截断点为position。cap为截断值

**cap = (totalCap - salariesSum[position-1]) // (len(salaries) -position )**

```
IF cap < salaries[position-1]
   表明截断值在当前截断点的前半部分
IF cap > salaries[position]
   表明截断值在当前截断点的后半部分
IF salaries[position-1]<=cap<=salaries[position]:
   return cap
```

### 9.4.3 代码实现

```python
class SalaryCap():
    def __init__(self, salaries, capTotal):
        self.salariesArray = salaries
        self.capTotal = capTotal
        self.salariesArray.sort()
        self.salasrySum = []
        sum = 0
        for i in self.salariesArray:
            sum += i
            self.salasrySum.append(sum)
        if self.capTotal > self.salasrySum[-1]:
            print('总薪资不能大于原来工资的综总和！')

    def getSalaryCap(self):
        b, e = 0, len(self.salariesArray) - 1
        while b <= e:
            m = b + (e - b) // 2
            remain = self.capTotal - self.salasrySum[m - 1]
            possibleCap = remain / (len(self.salariesArray) - m)
            if possibleCap < self.salariesArray[m - 1]:
                # 如果截断值比原来截断点前一个元素小，那么截断点应该在前半部分
                e = m - 1
            elif possibleCap > self.salariesArray[m]:
                # 如果截断值比该位置的元素小， 那么截断值应该在后半部分
                b = m + 1
            elif self.salariesArray[m - 1] <= possibleCap <= self.salariesArray[m]:
                return possibleCap
        return -1

import random
salaries = []
for i in range(10):
    salaries.append(random.randint(100,200))
print('原来员工工资：', salaries)
sort_salaries = sorted(salaries)
caposition = random.randint(0, len(salaries)-1)
cap = (sort_salaries[caposition-1] + sort_salaries[caposition]) / 2
print('当前的截断值为：', cap)
totalCap = sum(sort_salaries[0:caposition]) + cap * (len(sort_salaries)-caposition)
print('预测的截断值，', SalaryCap(salaries, totalCap).getSalaryCap())
'''
原来员工工资： [183, 110, 122, 168, 182, 159, 162, 183, 105, 102]
当前的截断值为： 175.0
预测的截断值， 175.0
'''
```

## 9.5在二维数组中快速查找给定值

### 9.5.1 题目描述

在一个行和列按升序排列并且前一行的最后一个元素小于下一行的第一个元素。

![1560610862315](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560610862315.png)

### 9.5.2  解题思路

目标值总是与一行的最后一个元素做对比。

![1560610997868](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560610997868.png)                                                                                            

### 9.5.3 代码实现

```python
def twoArrarySearch(nums, target):
    if len(nums)==0 and len(nums[0])==0:
        return False
    i, j = 0, len(nums[0])-1
    while i<len(nums) and j >=0:
        if nums[i][j]==target:
            return True
        elif nums[i][j] > target:
            j -= 1
        else:
            i += 1
    return False

A = [[2,4,6,8,10],
     [12,14,16,18,20],
     [22,24,26,28,30],
     [32,34,36,38,40],
     [42,44,46,48,50]
     ]
print(twoArrarySearch(A, 14))
```

## 9.6 快速在组合中查找重复元素和组合元素

### 9.6.1 题目描述

给定一个集合Z，它所包含的元素在1到n（包括n）之间且有元素之间不重合。再给定一个集合A，它所包含的元素有n个， 但是元素的范围是1到n-1，且含有一个重复元素。要求找到遗失的元素和重复的元素。集合Z和集合A的长度都等于n。

![1560649960178](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560649960178.png)

![1560650027016](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560650027016.png)

### 9.6.2 解题思路

第一步：求出两个集合的元素之和。sum(Z)，sum(A)。假设遗失元素x，重合元素为y。则有sum(Z) - sum(A) = x - y = s1

第二步：求出各个集合的元素平方和。sqrt_sum(Z) - sqrt_sum(A) = x2 - y2 = (x +y) (x - y)，s2 = x + y =( x2 - y2) / 2

第三步：求出s1和s2的值。x  =  (s1 + s2) /2,  y = (s2 - s1) /2

### 9.6.3 代码实现

```python
def searchMissAndRepeat(Z, A):
    if len(Z)==0 or len(A)==0:
        return None
    sum_z, sum_a = sum(Z), sum(A)
    sqrt_sum_z, sqrt_sum_a = 0, 0
    for i in range(len(Z)):
        sqrt_sum_a += A[i]*A[i]
        sqrt_sum_z += Z[i] * Z[i]
    s1 = sum_z - sum_a
    s2 = (sqrt_sum_z - sqrt_sum_a) / s1
    miss_num = int((s1 + s2) / 2)
    repeat_num = int((s2 - s1) / 2)
    return [miss_num, repeat_num]

Z = [1,2,3,4,5,6]
A = [1,2,3,4,4,5]
print('遗失的数字和重复的数字',searchMissAndRepeat(Z, A))
```

## 9.7 绝对值排序

### 9.7.1 题目描述

对一个只包含整数的数组，以绝对值的方式进行排序。

### 9.7.2 解题思路

使用python的匿名函数lambda来解决。

### 9.7.3 代码实现

```python
def abs_sort(nums):
    if len(nums)==0:
        return nums
    nums.sort(key=lambda x:abs(x))

nums = [-6, -1, 2, -3, 4, -5]
abs_sort(nums)
print(nums)  #[-1, 2, -3, 4, -5, -6]
```

## 11 贪婪算法

## 11.1 最小生成树（未做）

### 11.1.1 题目描述

给定一个边和带权重的无向连通图，找到一条路径连通所有节点，同时使得路径的权重和最小。

![1560504393573](C:\Users\gzr\AppData\Roaming\Typora\typora-user-images\1560504393573.png)

### 11.1.2 解题思路

### 11.3 代码实现

## 11.2 哈弗曼编码

### 11.2.3 代码实现

```python
# 定义哈夫曼树的节点属性
class HuffNode():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

# 每个节点都为叶节点，建立叶节点
def CreatNode(vals):
    nodes = []
    for i in range(len(vals)):
        newNode = HuffNode(vals[i])
        nodes.append(newNode)
    return nodes

# 建立霍夫曼树
def CreatHUffTree(nodes):
    queue = nodes.copy()
    while len(queue)>1:
        queue.sort(key=lambda x:x.val)
        left = queue.pop(0)
        right = queue.pop(0)
        root = HuffNode(left.val + right.val)
        root.left = left
        root.right = right
        left.parent = root
        right.parent = root
        queue.append(root)
    return queue[0]

# 生成霍夫曼编码
def CreatHUffCode(root, code, res):
    if root.left==None and root.right==None:
        res.append((root.val, code))
        return res
    if root.left:
        res = CreatHUffCode(root.left, code + '0', res)
    if root.right:
        res = CreatHUffCode(root.right, code + '1', res)
    return res

freqs = [35, 10, 20, 20, 15]
nodes = CreatNode(freqs)
root = CreatHUffTree(nodes)
dic = CreatHUffCode(root, '', [])
print(dic)
#[(20, '00'), (20, '01'), (10, '100'), (15, '101'), (35, '11')]
```

## 11.3 离散点的最大覆盖率

### 11.3.1 题目描述

所有元素的集合为U = {1,2,3,4,5}，S1 = {1,2,3}， S2= {2,4}，S3= {1,3,5}，S4= {4,5}，要求4个子集中选出最少的几个，使得它们能够覆盖U。

### 11.3.2 解决思路

首先选出最长的集合，然后在剩余集合中去除与当前所集合重合的元素。然后再选出集合最长的元素，以此类推，知道所选的集合中的总的元素等于U即可。

### 11.3.3 代码实现

```python
def solve(U, SubSet):
    queue = SubSet.copy()
    tmp_set = set()  # 只存放集合的元素
    res = []  # 存放最终的结果列表
    while len(queue) > 0:
        queue.sort(key=lambda x: len(x[1]), reverse=True)
        current_set = queue.pop(0)
        tmp_set.update(current_set[1])
        res.append(current_set)
        if tmp_set == U:
            break
        for v in queue:
            v[1] = v[1] - current_set[1].intersection(v[1])
    return res

U = {1, 2, 3, 4, 5}
dic = {'S1': {1, 2, 3}, 'S2': {2, 4}, 'S3': {1, 3, 5}, 'S4': {4, 5}}
subset = [list(i) for i in dic.items()]
print(solve(U, subset))
#[['S1', {1, 2, 3}], ['S4', {4, 5}]]
```

## 12 动态规划

## 12.1 钢管的最优切割方案

### 12.1.1 题目描述

给定钢管切割长度和相应的价格。

| 长度 *i*  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| --------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 价格 *pi* | 1    | 5    | 8    | 9    | 10   | 17   | 17   | 20   | 24   | 30   |

设计一个有效算法，使得切割出来的钢管卖出最高的价格。

### 12.1.2 解题思路

用n来表示钢管的长度，在选择最优切割方案时，将钢管从切割1米到n米的最优方案进行比较。

### 12.1.3 代码实现

```python
def BottomUpCutRod(p, n):
    if n==0:
        return 0
    r = [0]*(n+1)
    for i in range(1, n+1):
        tmp = 0
        for j in range(1, i+1):
            tmp = max(tmp, p[j]+r[i-j])  # 核心步骤
        r[i] = tmp
    return r[-1], r
p=[0,1,5,8,9,10,17,17,20,24,30]
print(BottomUpCutRod(p, 10))   #  (30, [0, 1, 5, 8, 10, 13, 17, 18, 22, 25, 30])
```

## 12.2 求两个字符串的最长公共子串

### 12.2.1 题目描述

找出两个字符串的最长公共子串。例如：‘abccade’与字符串'dgcadde'的最长公共子串为'cad'.

### 12.2.2 解决思路

使用暴力匹配，时间复杂度为len(s1)*len(s2) *len(最大子串)，空间复杂度为O(1)

使用动态规划，时间复杂度为len(s1)*len(s2)， 空间复杂度为len(s1) *len(s2) 

### 12.2.3 代码实现

暴力匹配

```python
def matchMaxString(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return None
    max_len = 0   # 存放子串的最长长度
    max_str = ''    # 存放最长子串
    for i in range(len(s1)):
        for j in range(len(s2)):
            c = 0   # 用与计数当前重合的字符
            if s1[i] == s2[j]:
                c = match(s1, i + 1, s2, j + 1, c + 1)
                if c > max_len:
                    max_len = c
                    max_str = s1[i:i + c]
    return max_str, max_len

def match(s1, i, s2, j, c):
    # 下标越界或者不重合
    if i >= len(s1) or j >= len(s2) or s1[i] != s2[j]:
        return c
    c = match(s1, i + 1, s2, j + 1, c + 1)
    return c

```

动态规划

```python
def matchMaxString_two(s1, s2):
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    max_len = 0        # 存放子串的最长长度
    max_str = ''       # 存放最长子串
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1  #　关键
                if dp[i + 1][j + 1] > max_len:　　　
                    max_len = dp[i + 1][j + 1]
                    max_str = s1[i - max_len + 1:i + 1]　 # 存放最长子串
    return max_str, max_len

s1, s2 = 'abccade', 'dgcadde'
print(matchMaxString_two(s1, s2))
```

