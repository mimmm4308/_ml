# 使用chatgpt 看過且懂了
import random

# 定義目標函數
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 爬山演算法參數
step_size = 0.1
max_iterations = 10000

# 初始化隨機起點
x, y, z = [random.uniform(-10, 10) for _ in range(3)]
current_value = f(x, y, z)

for i in range(max_iterations):
    # 嘗試往附近移動
    neighbors = []
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            for dz in [-step_size, 0, step_size]:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                neighbors.append((nx, ny, nz))

    # 找出最好的鄰居
    next_x, next_y, next_z = x, y, z
    best_value = current_value
    for nx, ny, nz in neighbors:
        value = f(nx, ny, nz)
        if value < best_value:
            best_value = value
            next_x, next_y, next_z = nx, ny, nz

    # 如果沒有更好的點就停止
    if best_value >= current_value:
        break

    # 更新當前位置
    x, y, z = next_x, next_y, next_z
    current_value = best_value

print(f"最小值位置：x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
print(f"對應的函數值：f(x, y, z) = {current_value:.4f}")
