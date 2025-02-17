# run on python 3.7.2
from sys import stdin


# 101010
# 100100100

def main():
    t = int(stdin.readline())
    for i in range(t):
        n = stdin.readline()
        s = stdin.readline()
        print(solve(s))

def solve(n):
    res = 0
    prev = 0
    n = n[:-1]
    for x in n:
        x = int(x)
        if x == 1 and prev == 0:
            res += 1
            prev = 1
        if x == 0 and prev == 1:
            res += 1
            prev = 0
    return(res)


main()
