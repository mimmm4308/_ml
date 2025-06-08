#使用chatgpt 看過且懂了
#分群
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# 讀資料
digits = load_digits()
X = digits.data
y = digits.target  # 這裡只有用來評估，不用在訓練

# 建立 KMeans 分群模型，設群數為10 (數字0-9)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

# 預測群集標籤
clusters = kmeans.labels_

# 評估：用 Adjusted Rand Index 衡量分群結果跟真實標籤的相似度
ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index: {ari:.3f}")
