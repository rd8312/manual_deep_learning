from typing import Any
from step01_variable import Variable
import numpy as np

# class Function:
#     ## __call__: Python 特殊方法，只要定義，當 f = Function()時，寫成 f(...)就可以呼叫出 __call__方法
#     def __call__(self, input):
#         x = input.data # 取出資料
#         y = x**2 # 實際計算
#         output = Variable(y) # 回傳 Variable
#         return output


# 把 Function 類別作為基礎，以繼承此類別來執行 Function 類別
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 利用 forward 方法進行具體運算
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x **2


if __name__ == "__main__":
    # x = Variable(np.array(10))
    # f = Function()
    # y = f(x)

    # print(type(y))
    # print(y.data)

    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)
