#使用chatgpt 看過且懂了
"""
什麼是 Gradient Boosting？

Gradient Boosting 是一種集成學習(Ensemble Learning)方法，透過串聯多個弱學習器（通常是決策樹），
每一個新模型都是針對前一個模型錯誤的部分進行「梯度下降」式的優化，最終合成一個強學習器。

簡單來說：
- 先用一棵樹做預測
- 觀察預測誤差（殘差）
- 用另一棵樹去學習修正這些誤差
- 不斷疊加多棵樹，提升預測準確度

Gradient Boosting 原理簡述：

1. 初始化模型 F_0(x)，通常是一個常數（如平均值）

2. 對於每個 boosting 步驟 m=1,2,...,M：
   - 計算目前模型的負梯度（殘差）
     r_im = - [∂L(y_i, F(x_i)) / ∂F(x_i)]_{F=F_{m-1}} ，這代表對損失函數的負梯度
   - 擬合一個弱學習器 h_m(x) 去預測負梯度 r_im
   - 更新模型：F_m(x) = F_{m-1}(x) + ν h_m(x)，其中 ν 是學習率（通常小於1）
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 載入資料
iris = load_iris()
X, y = iris.data, iris.target

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立 Gradient Boosting 分類器
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 訓練模型
gb_clf.fit(X_train, y_train)

# 預測
y_pred = gb_clf.predict(X_test)

# 輸出準確率
print("Accuracy:", accuracy_score(y_test, y_pred))
