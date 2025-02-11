# run on python 3.7.2
from sys import stdin

# x, y
# n
# 1, 2
# 1 2
# 77, 77
# 999 1
# 9 * 111
# 18 * 1
# 9*10**z * 1
# 99
# 77 / 11 = 7
# 7 * 11
# 77777777777 
# S(n + 1) = S(n) + 1 if no change of 10
# S(n + 1) = 1 if all the digits of n are 9

# 11119 S(n) = 13
# 11120 S(n+1) = 5 (-9 + 1)
# xx999 S(n+1) = S(n) + -9 * numnber_of_ 9 from the right + 1
# 22999 31
# 23000 5

# y = x + -9 * 9 digits of N + 1
# S(n) = x

# 13
# 9 + 1 * 4
# 8 + 2 + 1 * 3
# 82111
# 11128
# S(n) = x = 13
# S(n + 1) = y = 14

# if y = x + 1
# if y = 1, x = 

# 1 11
# x=36
# y=5
# 10000
# 10001
# 00010
# 9 9


def main():
    n = int(stdin.readline())
    for i in range(n):
        x, y = map(int, stdin.readline().split())
        solve(x,y)
def solve(x,y):
    if y == x + 1:
        print("Yes")
        return
    if x > y and ((x - y) + 1)  % 9 == 0:
        print("Yes")
        return
    print("No")



main()
