import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 取得函數
            x, y = f.input, f.output # 取得函數的輸出入
            x.grad = f.backward(y.grad) # 呼叫backward方法

            if x.creator is not None:
                funcs.append(x.creator) # 在清單加入上一個函數

# reference: test.ipynb step 9
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 利用 forward 方法進行具體運算
        output = Variable(as_array(y))
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
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)
    

if __name__ == "__main__":
    # 反向傳播
    x = Variable(np.array(0.5))
    y = square(exp(square(x))) # 連續套用
    y.backward()
    print(x.grad)

    x = Variable(np.array(1.0)) # OK
    x = Variable(None) # OK

    x = Variable(1.0) # NG : 發生錯誤!
