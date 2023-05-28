import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 1. 取得函數
        if f is not None:
            x = f.input # 2. 取得函數的輸入
            x.grad = f.backward(self.grad) # 3. 呼叫函數的backward方法
            x.backward() # 呼叫上一個變數的backward方法 (遞迴)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 利用 forward 方法進行具體運算
        output = Variable(y)
        output.set_creator(self) # 讓輸出變數記住它的生身父母
        self.input = input # 記住輸入變數，為了反向傳播可以計算
        self.output = output # 也記住輸出
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 反向回溯計算圖的節點
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # 反向傳播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
