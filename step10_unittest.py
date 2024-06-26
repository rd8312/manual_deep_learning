import unittest
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
    
def square(x):
    return Square()(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    # test 1
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        
        expected = np.array(4.0)
        
        self.assertEqual(y.data, expected)

    # test 2
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        
        expected = np.array(6.0)
        
        self.assertEqual(x.grad, expected)

    # test 3
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()

        num_grad = numerical_diff(square, x)

        # np.allclose(a, b, rtol=1e-05, atol=1e-08)
        # if |a-b| <= atol + rtol*|b|, return True
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
    

unittest.main()
