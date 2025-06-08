#使用chatgpt 看過且懂了
import random
import math

# 城市座標
cities = [
    (0,3), (0,0), (0,2), (0,1),
    (1,0), (1,3), (2,0), (2,3),
    (3,0), (3,3), (3,1), (3,2)
]

# 計算兩城市之間距離
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# 計算路徑總距離
def total_distance(path):
    dist = 0
    for i in range(len(path)):
        dist += distance(cities[path[i]], cities[path[(i+1) % len(path)]])
    return dist

# 產生鄰近解：隨機交換兩個城市
def get_neighbor(path):
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# 爬山演算法主程序
def hill_climb_tsp(cities, max_iterations=10000):
    current_path = list(range(len(cities)))
    random.shuffle(current_path)
    current_distance = total_distance(current_path)

    for _ in range(max_iterations):
        neighbor = get_neighbor(current_path)
        neighbor_distance = total_distance(neighbor)

        if neighbor_distance < current_distance:
            current_path = neighbor
            current_distance = neighbor_distance

    return current_path, current_distance

# 執行演算法
best_path, best_dist = hill_climb_tsp(cities)
print("最佳路徑:", best_path)
print("總距離:", round(best_dist, 2))
