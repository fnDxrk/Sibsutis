import numpy as np
def rukzak(M, price : dict):
    # price словарь
    # m масса отдельного предмета
    # M предельная масса
    # c стоимость предмета
    n = len(price["m"])
    C = np.zeros(n)
    for i in range(n):
        if M >= price["m"][i]:
            # массив в котором каждый эмемент отображает сколько PRICEEEE 
            # будет при выборе этого элемента
            C[i] = rukzak(M - price["m"][i], price) + price["c"][i]
    return np.max(C)

if __name__ == "__main__":
    price = {
        "m" : [3, 5, 8],
        "c" : [8, 14, 23]
    }
    for i in range(14):
        print (f"f({i}) = {rukzak(i, price)}")
