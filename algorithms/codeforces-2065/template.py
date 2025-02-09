# run on python 3.7.2
from sys import stdin


def main():
    n = int(stdin.readline())
    s = stdin.readline()
    x = [None] * n
    y = [None] * n
    for i in range(n):
        x[i], y[i] = map(int, stdin.readline().split())
    # solve here
    print(n)
    print(s)
    print(x[0])
    print(y[n - 1])


def solve(n):
    return 0


main()
