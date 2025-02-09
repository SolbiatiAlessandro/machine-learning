# run on python 3.7.2
from sys import stdin

# 2 rows with m seats each
# a + b + c
# a -> 1
# b -> 2
# c -> ?

# m,a,b,c
# 10, 5, 5, 10
# 3, 6, 1, 1
# 15, 15, 12, 14



def main():
    n = int(stdin.readline())
    for i in range(n):
        m, a, b, c = map(int, stdin.readline().split())
        solve(m,a,b,c)


def solve(m,a,b,c):
    r1 = min(m, a) 
    r2 = min(m, b) 
    left = min(m - r1 + m - r2, c)
    print(r1 + r2 + left)


main()
