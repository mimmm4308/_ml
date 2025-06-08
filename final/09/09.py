#使用chatgpt 看過且懂了
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 準備訓練資料 (x, y)
# 假設 y = 2x + 3 加上一些噪音
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = torch.tensor([[5.0], [7.0], [9.0], [11.0], [13.0]])

# 2. 定義線性回歸模型
model = nn.Linear(in_features=1, out_features=1)

# 3. 定義損失函數 (均方誤差) 和優化器 (SGD)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 訓練模型
epochs = 1000
for epoch in range(epochs):
    model.train()
    
    # 預測
    outputs = model(x_train)
    
    # 計算損失
    loss = criterion(outputs, y_train)
    
    # 梯度歸零
    optimizer.zero_grad()
    
    # 反向傳播
    loss.backward()
    
    # 更新權重
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 印出訓練後模型參數
[w, b] = model.parameters()
print(f'訓練後的權重: {w.item():.4f}, 偏差: {b.item():.4f}')

# 6. 測試模型 (舉例輸入6)
model.eval()
with torch.no_grad():
    x_test = torch.tensor([[6.0]])
    y_pred = model(x_test)
    print(f'測試輸入6得出模型預測結果為: {y_pred.item():.4f}')
