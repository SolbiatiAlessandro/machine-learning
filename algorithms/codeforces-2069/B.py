# run on python 3.7.2
import sys

# greedy
# pick always the largest set you can?

# need to pick the diagonals
# but how do I pick the target

# if a number share side with same number , number is in both set
# first set
# second set
# Q: are there only two sets? yes


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    t = data[0]
    idx = 1
    results = []
    
    for _ in range(t):
        n, m = data[idx], data[idx+1]
        idx += 2
        a = [data[idx + i*m : idx + (i+1)*m] for i in range(n)]
        idx += n * m
        results.append(str(solve(a, n, m)))
    
    sys.stdout.write("\n".join(results))

from itertools import product
from collections import defaultdict

def solve(a, n, m, debug=True):
    #print(a)
    set1 = defaultdict(int)
    set2 = defaultdict(int)
    lenset1 = 0
    lenset2 = 0

    def f(ix):
        nonlocal lenset1, lenset2
        i,j = ix
        x = a[i][j]
        if set1[x] == 0:
            set1[x] = 1
            lenset1 += 1

        if set2[x] == 0:
            n1, n2 = -1, -1
            if (i + 1) < n:
                n1 = a[i+1][j]
            if (j + 1) < m:
                n2 = a[i][j+1]
            if n1 == x or n2 == x:
                set2[x] = 1
                lenset2 += 1

    list(map(f, product(range(n), range(m))))

    res = -1 if lenset2 == 0 else -2
    res += lenset1 + lenset2
    return res

main()
