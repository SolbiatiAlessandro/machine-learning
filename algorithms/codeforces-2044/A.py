# run on python 3.7.2
from sys import stdin


# a = n - b
# 2 
# n = a + b


def main():
    n = int(stdin.readline())
    for _ in range(n):
        s = int(stdin.readline())
        solve(s)


def solve(n):
    print(n - 1)

main()
