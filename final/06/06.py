#使用chatgpt 看過且懂了
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data  # 數值
        self.grad = 0.0   # 對目標的導數 (反向傳播後會被填入)
        self._backward = lambda: None  # 用於反向傳播的函數
        self._prev = set(_children)    # 前驅節點
        self._op = _op                 # 操作名稱（例如 '+', '*', 'exp'）
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def exp(self):
        # 計算 e^x
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # 導數為 e^x，也就是 out.data
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        # 計算 1 / (1 + e^(-x))，常見於神經網路
        x = self.data
        sigmoid_val = 1 / (1 + math.exp(-x))
        out = Value(sigmoid_val, (self,), 'sigmoid')

        def _backward():
            # 導數為 sigmoid(x) * (1 - sigmoid(x))
            self.grad += sigmoid_val * (1 - sigmoid_val) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # 拓撲排序 + 反向傳播
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
#範例
x = Value(2.0)
y = x.exp()
y.backward()
print(f"exp(2.0) = {y.data}, dy/dx = {x.grad}")

x = Value(0.0)
y = x.sigmoid()
y.backward()
print(f"sigmoid(0.0) = {y.data}, dy/dx = {x.grad}")
