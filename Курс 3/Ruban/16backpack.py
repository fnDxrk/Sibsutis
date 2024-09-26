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
            # массив в котором каждый эмемент отображает сколько деняг отбашлять
            # надо при выборе этого элемента
            C[i] = rukzak(M - price["m"][i], price) + price["c"][i]
    return np.max(C)

price = {
    "m" : [3, 5, 8],
    "c" : [8, 14, 23]
}
M = 15
for i in range(M+1):
    print (f"f({i}) = {rukzak(i, price)}")

print(f"Максимальная стоимость набора товаров при грузоподъемности {M}: {rukzak(i, price)}")
