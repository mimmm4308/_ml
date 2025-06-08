#使用chatgpt 看過且懂了
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        # 實際的數值
        self.data = data
        # 對某個最終輸出（通常是損失函數）的導數
        self.grad = 0.0
        # 儲存這個值是怎麼來的（從哪些前一個 Value 算出來的）
        self._prev = set(_children)
        # 對應的操作（例如 '+', '*', '**2', 'exp' 等）
        self._op = _op
        # 反向傳播函數的預設為空操作，會在每個運算中被定義
        self._backward = lambda: None

    def __add__(self, other):
        # 支援 Value + 常數
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # 定義這個加法對兩邊的偏導為 1
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other): return self + other

    def __mul__(self, other):
        # 支援 Value * 常數
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        # 對於乘法：∂(a*b)/∂a = b，∂(a*b)/∂b = a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other

    def __neg__(self): return self * -1  # 支援 -Value

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return other + (-self)

    def __truediv__(self, other): return self * other**-1  # 支援除法

    def __pow__(self, power):
        # 次方操作，僅支援數值次方
        assert isinstance(power, (int, float)), "次方只能是數字"
        out = Value(self.data ** power, (self,), f'**{power}')
        def _backward():
            # 導數公式為 power * x^(power - 1)
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        # e^x 函數，常見於機器學習中
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad  # d/dx(e^x) = e^x
        out._backward = _backward
        return out

    def sigmoid(self):
        # Sigmoid 函數：f(x) = 1 / (1 + e^-x)
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad  # 導數為 f(x)*(1-f(x))
        out._backward = _backward
        return out

    def backward(self):
        # 執行反向傳播來計算所有節點的梯度
        topo = []      # 拓撲排序後的節點順序（先計算的放後面）
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0  # 損失對自己的導數設為 1
        for node in reversed(topo):
            node._backward()  # 自下而上依序計算導數

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# 初始化變數（任意值都可以）
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 設定學習率
learning_rate = 0.1

# 梯度下降主迴圈
for step in range(100):
    # 每次迴圈前重設梯度為 0
    x.grad = y.grad = z.grad = 0.0

    # 定義損失函數 f(x, y, z)
    # f = x^2 + y^2 + z^2 - 2x - 4y - 6z + 8
    f = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

    # 執行反向傳播來計算每個變數的梯度
    f.backward()

    # 使用梯度下降更新變數
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad

    # 每 10 步列印一次進度
    if step % 10 == 0:
        print(f"step {step:02d} | f = {f.data:.4f} | x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")

# 印出最終結果
print("\n最低點結果")
print(f"x = {x.data:.4f}, y = {y.data:.4f}, z = {z.data:.4f}")
