# run on python 3.7.2
from sys import stdin

"""
m=1
a,b

n,m
i [1,n]
j = 1
a = b - a

4 1
5 4 10 5
5, -1 + 4, 0 + 10, -6 + 5, -1
4
-1 

9 8 7
8

1 4 3
3

1, 2 + 4, -1, 

9, -1 - 8, 0, - 7 , 1

-1 0 7

1 4 2 5

5 2
6 4 5 4 5
4 1000
"""




def main():
    t = int(stdin.readline())
    x = [None] * t
    y = [None] * t
    for i in range(t):
        n, m = map(int, stdin.readline().split())
        a  = list(map(int, stdin.readline().split()))
        b  = list(map(int, stdin.readline().split()))
        solve(n, m, a, b)


def solve(n, m, a, b):
    b0 = b[0] # 4
    possible = [a[0], b0 - a[0]] # 5, -1
    for i, ai in enumerate(a): # 1, 4
        #print(possible, i, ai)
        n_possible = []
        x = ai
        if x >= possible[0] or (len(possible) > 1 and x >= possible[1]):
            n_possible.append(x)
        x = b0 - ai
        if x >= possible[0] or (len(possible) > 1 and x >= possible[1]):
            n_possible.append(x)
        if len(n_possible) == 0:
            print("NO")
            return
        possible = n_possible
    print("YES")





main()
