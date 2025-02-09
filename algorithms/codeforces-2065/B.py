# run on python 3.7.2
from sys import stdin

# s
# s > 1
"""
0 < i < len(s)
s[i] = s[i+1]
s[i] = rand(lower)
s.drop(i + 1)

baa

aabcc

aabcckhf
aabkkhf
aabhhf
aabff
aabb
aaa

skibidus

"""


def main():
    n = int(stdin.readline())
    for i in range(n):
        s = stdin.readline()
        solve(s)

def solve(s):
    s = s[:-1]
    for i, nc in enumerate(s[1:]):
        c = s[i] 
        if c == nc:
            print(1)
            return
    print(len(s))
        



main()
