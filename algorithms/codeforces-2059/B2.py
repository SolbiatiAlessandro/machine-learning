# run on python 3.7.2
from sys import stdin


def solve(a, n, m):
    # 1 1 2 2 3 3 4 4
    if n == m:
        for i in range(n): # 1
            if i % 2 == 1: 
                b_index = (i // 2) + 1 # 1
                if b_index != a[i]: # 1
                    return b_index
        return b_index + 1
    
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
