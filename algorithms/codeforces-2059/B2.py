# run on python 3.7.2
from sys import stdin


def solve(a, n, m):
   # 1 1 1 10k 2, 5, 3
    if n == m:
        full = True
        for i in range(int(n / 2)):
            if (i + 1) != a[i*2]:
                full = False
        if full:
            return int(n/2) + 1
    
    gap = n - m # 1
    for k in range(gap + 1): 
        b1_end_index = k+1 # 2
        #b1 = a[0:b1_end_index]
        b2_start = a[b1_end_index] # 1
        if b2_start != 1:
            return 1
    return 2



def main():
    t = int(stdin.readline())
    for i in range(t):
        m, n = tuple(map(int, stdin.readline().split()))
        a = list(map(int, stdin.readline().split()))
        res = solve(a, m, n)
        print(res)


main()
