# 实现一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

if __name__ == '__main__':
    a = int(input('a = '))
    b = int(input('b = '))
    print('%d + %d = %d' % (a, b, add(a, b)))
    print('%d - %d = %d' % (a, b, sub(a, b)))
    print('%d * %d = %d' % (a, b, mul(a, b)))
    print('%d / %d = %d' % (a, b, div(a, b)))

# 实现一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

if __name__ == '__main__':
    a = int(input('a = '))
    b = int(input('b = '))
    print('%d + %d = %d' % (a, b, add(a, b)))