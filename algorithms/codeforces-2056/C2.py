# run on python 3.7.2j
import bisect
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
4 10 11 12 13 1000

i = 0
min 4 - 6 = -2 
max 1000 - 6 = 994

i = 1
min 4 - 4 = 0
max 1000 - 4 = 996
"""




def main():
    t = int(stdin.readline())
    x = [None] * t
    y = [None] * t
    for i in range(t):
        n, m = map(int, stdin.readline().split())
        a  = list(map(int, stdin.readline().split()))
        b  = list(map(int, stdin.readline().split()))
        solve(n, m, a, b, _print=True)


def solve_TLE(n,m,a,b, _print=False):
    #print(a, b)
    b0 = b[0] # 4
    possible = [a[0]]
    for b0 in b:
        possible.append(b0 - a[0])
    for i, ai in enumerate(a): # 1, 4
        #print(possible, i, ai)
        possible = sorted(possible)
        n_possible = [ai]
        for b0 in b:
            n_possible.append(b0 - ai)

        n_possible = sorted(n_possible)


        i = 0
        for curr in n_possible:
            if not curr >= possible[0]:
                i += 1
        n_possible = n_possible[i:]
        #print(f"{i}, previous possible {possible}, current possible {n_possible}")


        if len(n_possible) == 0:
            if _print:
                print("NO")
            else:
                return 0
            return
        possible = n_possible
    if _print:
        print("YES")
    else: 
        return 1

def solve(n, m, a, b, _print=False, debug=False):
    """
    n:4 m:3
    a: 2 4 6 5
    b: 6 1 8
    """
    b = sorted(b) #1, 6, 8
    maxes = [None] * n #[n,n,n,n]
    maxes[n - 1] = max(b[-1] - a[n - 1], a[n - 1]) #[n,n,2,5]
    i = n - 2 # 3
    while i >= 0: # 1
        if debug: print(f"{i}, maxes: {maxes}")
        
        previous_max = maxes[i + 1] # 2
        current_a = a[i] # 4

        # b: 4, 1000
        # b - a: 0, 996
        # [b - a] + [a]: 0, 4, 996
        #current_ai = bisect.bisect_left(b, current_a) # 

        #print("sorted b before a insertion:", b)
        #b_with_current_a = b[:current_ai] + [current_a] + b[current_ai:]
        #print("sorted b after a insertion:", b_with_current_a)


        search = previous_max + current_a # 2 + 4 = 6
        current_bi = bisect.bisect_right(b, search) # 1
        if debug: print(f"{i}: searched {search} , bisect at index {current_bi}")

        current_max = -10e10
        if current_bi > 0:
            best_b = b[current_bi - 1] # 4
            current_max = max(best_b - current_a, current_max)

        if current_a <= previous_max:
            current_max = max(current_a, current_max)


        if current_max == -10e10:
            if _print:
                print("NO")
                return
            else:
                return 0

        if debug: print(f"{i}: b {b}, current_bi to insert at {current_bi}")

        maxes[i] = current_max # [n,n,n,0,995]
        i -= 1
    if debug: print(f"{i}, maxes: {maxes}")
    if _print:
        print("YES")
        return
    else:
        return 1

from random import random
def test_match(ix):
    n, m = int(random()*10) + 1, int(random()*10) + 1
    a = [int(random()*20) + 1 for _ in range(n)]
    b = [int(random()*20) + 1 for _ in range(n)]


    passed = solve(n,m,a,b) == solve_TLE(n,m,a,b)
    if not passed:
        print("WRONG ANSWER")
        print(f"n={n}")
        print(f"m={m}")
        print(f"a={a}")
        print(f"b={b}")
    else: print(f"{ix} PASSED")

def test_wrong_answers():
    n=8
    m=9
    a=[1, 13, 11, 10, 13, 4, 2, 1]
    b=[18, 16, 1, 16, 9, 8, 12, 13]
    assert solve(n,m,a,b,debug=False)

    n=6
    m=8
    a=[2, 8, 2, 2, 13, 18]
    b=[2, 14, 19, 14, 3, 7]
    assert solve(n,m,a,b,debug=True,_print=True)


#if __name__ == "__main__":
#    test_wrong_answers()
#    for i in range(10): test_match(i)
    




    
    """
    while i > 0:
        mi, ma = p[i]
        pmi, pma = p[i-1]
        if ma < pmi:
            print("NO")
            return 
        p[i-1][1] = min(ma, pma)
    print("YES")
    return



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
    """





main()
