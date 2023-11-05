import numpy as np

# python編碼規則PEP8: 類別名稱的第一個字母為大寫
class Variable:
    def __init__(self, data):
        self.data = data

if __name__ == "__main__":
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)

    x.data = np.array(2.0)
    print(x.data)
