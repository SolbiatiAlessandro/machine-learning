# run on python 3.7.2
from sys import stdin

# 2,2,3 -> 2
# mode = max freq
# a, n
# b, n 
"""
for i in range:
    a[i] = max_freq(b[:i])

max_freq i = f(max freq i-1)

4
4
1 1 1 2
1 1 2 2
8
4 5 5 5 1 1 2 1

4 

4 5 5 1 1 2 2 1 
4 5 1 2 

freq = {4: 1}
max_freq = 1
a = 5
freq[5] = 0
need to insert 1 - 0 #5 
"""



def main():
    t = stdin.readline()
    for i in range(int(t)):
        n = int(stdin.readline())
        a = map(int, stdin.readline().split())
        present = [0] * n
        res = []
        for x in a:
            x = int(x)
            if not present[x - 1]:
                present[x - 1] = 1
                res.append(x)
        for i, x in enumerate(present):
            if x == 0:
                res.append(i + 1) 
        print(" ".join([str(x) for x in res]))


def solve(n):
    return 0


main()
