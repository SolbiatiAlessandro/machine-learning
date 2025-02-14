# run on python 3.7.2
from sys import stdin

# good = count(x) == 2, for x in a
# a, b 
# ci = ai + bi
# true, false

#4
#1 2 1 2
#1 2 1 2
"""
2 4 2 4

1 2 3 3 2 1
1 1 1 1 1 1

100 1 100 1
2 2 2 2

1000 * 2500

a a a
b1 b2 b3
if array unique value >= 3 yes
a a a
b b1 b1 
"""



def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = map(int, stdin.readline().split())
        b = map(int, stdin.readline().split())
        sa, sb = set(a), set(b)
        if len(sa) + len(sb) <= 3: print("NO")
        else: print("YES")
        #if len(sa) > 2: return True
        #if len(sb) > 2: return True
        #if len(sa) == len(sb) == 2: return True


def solve(n):
    return 0


main()
