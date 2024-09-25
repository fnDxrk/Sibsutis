def multiply_strings(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"

    n1, n2 = len(num1), len(num2)

    result = [0] * (n1 + n2)

    for i in range(n1 - 1, -1, -1):
        for j in range(n2 - 1, -1, -1):
            mul = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]

            result[p2] = total % 10
            result[p1] += total // 10

    result_str = ''.join(map(str, result)).lstrip('0')
    return result_str if result_str else "0"

num1 = "123"
num2 = "456"
print(multiply_strings(num1, num2)) 
