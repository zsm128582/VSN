下面给出一种贪心算法的 Python 实现方案，该算法的主要思路是：

1. **排序准备**  
   - 将所有商品价格按从小到大排序。  
   - 将优惠券按使用门槛 \(a_j\)（即最低使用金额）从小到大排序。

2. **贪心匹配**  
   - 遍历每个商品，假设当前商品价格为 \(p\)。
   - 对于所有使用门槛不高于 \(p\) 的优惠券，我们将它们加入一个“候选队列”（使用 Python 内置的堆，注意 Python 的 `heapq` 是最小堆，所以我们存入负的 \(b_j\) 值，以便取出最大折扣）。
   - 对于当前商品，从候选队列中取出折扣最大的优惠券（即拥有最大 \(b_j\) 的优惠券）并将其分配给该商品，从而使该商品获得最大的减免效果。
   - 累计所有使用的优惠券的减免金额得到最大总减免。

3. **时间复杂度**  
   - 由于商品和优惠券都只排序一次，并且每个优惠券只会进入堆一次，所以总体复杂度为 \(O((n+m) \log m)\) ，适用于 \(n, m \le 100\,000\)。

下面是完整的 Python 代码实现：

```python
import sys
import heapq

def main():
    # 读取输入数据
    data = sys.stdin.read().split()
    n, m = int(data[0]), int(data[1])
    
    # 读取商品价格列表
    prices = list(map(int, data[2:2+n]))
    
    # 读取优惠券信息 (a_j, b_j)
    coupons = []
    index = 2 + n
    for _ in range(m):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        coupons.append((a, b))
    
    # 商品价格和优惠券按照使用门槛进行排序
    prices.sort()
    coupons.sort(key=lambda x: x[0])
    
    total_discount = 0
    heap = []  # 用于存储候选优惠券的 -b_j 值（用最小堆模拟最大堆）
    coupon_index = 0

    # 遍历每个商品
    for price in prices:
        # 将所有当前可用的优惠券加入堆中
        while coupon_index < m and coupons[coupon_index][0] <= price:
            # 添加优惠券的折扣值 (注意存入负值)
            heapq.heappush(heap, -coupons[coupon_index][1])
            coupon_index += 1
        
        # 如果堆中有可用的优惠券，则选择折扣最大的那张
        if heap:
            total_discount += -heapq.heappop(heap)
    
    print(total_discount)

if __name__ == "__main__":
    main()
```

### 代码说明

- **输入处理与排序**  
  使用 `sys.stdin.read()` 将输入全部读取后，拆分为数字列表。前两个数字表示商品数量 \(n\) 和优惠券数量 \(m\)，接着是 \(n\) 个商品价格，然后是 \(m\) 组优惠券数据。  
  对商品价格列表使用 `prices.sort()` 进行排序，对优惠券使用 `coupons.sort(key=lambda x: x[0])` 按门槛排序。

- **贪心匹配策略**  
  遍历排序后的商品，对于每个商品，把所有满足优惠券门槛条件的优惠券加入堆中，然后从堆中取出折扣最大的优惠券分配给当前商品，累加到 `total_discount` 中。

- **输出**  
  最后打印累计的最大总减免金额。

该算法能够高效地解决问题，并确保总减免金额最大。