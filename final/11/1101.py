#使用chatgpt 看過且懂了
#分類
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 讀資料
data = load_wine()
X = data.data
y = data.target

# 切訓練集、測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立 SVM 分類器
model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))
