#使用chatgpt 看過且懂了
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 七段顯示器目標輸出（每行是7維表示）
y_true = np.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1],  # 9
])

# One-hot 輸入向量：數字 0~9
X = np.eye(10)

# 初始化權重與偏差
W = np.random.randn(10, 7) * 0.01
b = np.zeros((1, 7))

# 設定訓練參數
lr = 0.1
epochs = 1000
losses = []  # 儲存每次的損失以供繪圖

# 訓練迴圈
for epoch in range(epochs):
    # 前向傳播
    y_pred = X @ W + b
    y_pred_sigmoid = 1 / (1 + np.exp(-y_pred))

    # 計算損失 (MSE)
    loss = np.mean((y_pred_sigmoid - y_true) ** 2)
    losses.append(loss)

    # 計算梯度
    grad_output = 2 * (y_pred_sigmoid - y_true) / y_true.size
    grad_sigmoid = y_pred_sigmoid * (1 - y_pred_sigmoid)
    delta = grad_output * grad_sigmoid

    grad_W = X.T @ delta
    grad_b = np.sum(delta, axis=0, keepdims=True)

    # 更新權重與偏差
    W -= lr * grad_W
    b -= lr * grad_b

    # 每100輪印出一次損失
    if epoch % 100 == 0:
        print(f"第 {epoch} 輪，損失值 Loss: {loss:.5f}")

# ===== 測試預測結果 =====
print("\n=== 預測結果（四捨五入後）===")
pred = np.round(1 / (1 + np.exp(-(X @ W + b))))
correct = 0

for i in range(10):
    expected = y_true[i]
    predicted = pred[i].astype(int)
    match = np.array_equal(expected, predicted)
    correct += match
    print(f"數字 {i}: 預測={predicted}, 正確={expected}, 是否正確={match}")

# 顯示整體準確率
accuracy = correct / 10 * 100
print(f"\n整體預測準確率：{accuracy:.1f}%")

# ===== 繪製訓練損失曲線 =====
plt.plot(losses)
plt.title("訓練損失曲線")
plt.xlabel("訓練輪數")
plt.ylabel("損失 (MSE)")
plt.grid(True)
plt.show()
