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


class Function:
    def __call__(self, *inputs): # 改善1:可以不使用清單，給予任意數量的引數
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 改善2: 加註星號解封
        if not isinstance(ys, tuple): # 改善2: 增加處理非元祖的情況
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        # 改善1:當 outputs 的 element 只有一個，就回傳最初的元素
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y


def add(x0, x1):
    return Add()(x0, x1)


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1) # 自動轉成 (x0, x1) 作為 inputs
print(y.data)
