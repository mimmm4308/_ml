#使用chatgpt 看過且懂了
import sympy as sp

# 定義符號變數
x, y = sp.symbols('x y')

# ① f = x^2 * y
f1 = x**2 * y
grad_f1 = [sp.diff(f1, var) for var in (x, y)]

# ② f = sin(x) + y^2
f2 = sp.sin(x) + y**2
grad_f2 = [sp.diff(f2, var) for var in (x, y)]

# 輸出結果
print("① f = x^2 * y 的梯度為：")
print(f"∂f/∂x = {grad_f1[0]}, ∂f/∂y = {grad_f1[1]}")
print("\n② f = sin(x) + y^2 的梯度為：")
print(f"∂f/∂x = {grad_f2[0]}, ∂f/∂y = {grad_f2[1]}")
