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

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# old function
# class Function:
#     def __call__(self, input):
#         x = input.data
#         y = self.forward(x) # 利用 forward 方法進行具體運算
#         output = Variable(as_array(y))
#         output.set_creator(self) # 讓輸出變數記住它的生身父母
#         self.input = input # 記住輸入變數，為了反向傳播可以計算
#         self.output = output # 也記住輸出
#         return output
    
#     def forward(self, x):
#         raise NotImplementedError()
    
#     def backward(self, gy):
#         raise NotImplementedError()
    
class Function:
    def __call__(self, inputs):
        # inputs: the list of Variable objects
        xs = [x.data for x in inputs]  # Get data from Variable
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]  # Wrap data

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        print("result:", (y,))
        print("type:", type((y,)))
        print("type:", type(y,))
        return (y,) # return tuple
    
xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
print(ys)
print("ys type: ", type(ys))
y = ys[0]
print(y.data)
