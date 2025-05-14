MOD = 1000000007

import sys

def spiral_order(matrix, N):
    result = []
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    while top <= bottom and left <= right:
        # 从左到右
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        # 从上到下
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            # 从右到左
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            # 从下到上
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result

# 树状数组相关函数
def bit_update(bit, index, n):
    while index <= n:
        bit[index] += 1
        index += index & (-index)

def bit_query(bit, index):
    s = 0
    while index > 0:
        s += bit[index]
        index -= index & (-index)
    return s


if __name__ == "__main__":
    import sys
    # 调整递归深度，看需要添加
    sys.setrecursionlimit(10**6) 
    N = int(sys.stdin.readline())
    matrix = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # 获取矩阵的螺旋顺序数组
    spiral = spiral_order(matrix, N)
    total = N * N
    
    # 初始化树状数组 (1-indexed)
    bit = [0] * (total + 1)
    
    inv_count = 0
    seen = 0
    # 遍历螺旋顺序数组中的每个元素
    # BIT 操作都用 1-indexed 下标，且元素取值范围正好为 1~N^2
    for x in spiral:
        # 查询当前之前已经出现的元素中，小于等于 x 的数量
        cnt = bit_query(bit, x)
        # 当前比 x 大的个数 = 已出现的个数 - (小于等于x的个数)
        inv_count = (inv_count + seen - cnt) % MOD
        # 更新 BIT，标记 x 已出现
        bit_update(bit, x, total)
        seen += 1

    print(inv_count % MOD)

