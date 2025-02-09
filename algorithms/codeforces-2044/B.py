# run on python 3.7.2
from sys import stdin

# wqw
# wwqq -> ppww


def main():
    n = int(stdin.readline())
    for i in range(n):
        s = stdin.readline()
        solve(s)


def solve(s):
    res = []
    d = {'w':'w','p':'q','q':'p', '\n': ''}
    for c in s:
        res.insert(0, d[c])
    print(''.join(res))
    


main()
