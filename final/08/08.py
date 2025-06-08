#使用chatgpt 看過且懂了
import torch

# 建立變數 x, y, z，並設為可訓練（requires_grad=True）
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

# 使用 SGD 優化器（也可以用 Adam）
optimizer = torch.optim.SGD([x, y, z], lr=0.1)

# 執行多輪梯度下降
for step in range(100):
    optimizer.zero_grad()  # 歸零梯度，避免累加

    # 定義函數 f(x, y, z)
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    f.backward()  # 自動微分計算 df/dx, df/dy, df/dz

    optimizer.step()  # 更新變數（梯度下降）

    if step % 10 == 0:
        print(f"step {step:02d} | f = {f.item():.4f} | x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")

# 最終結果
print("\n最低點結果")
print(f"x = {x.item():.4f}, y = {y.item():.4f}, z = {z.item():.4f}")
