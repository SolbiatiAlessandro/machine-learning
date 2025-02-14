# run on python 3.7.2
from sys import stdin

"""
n
a
k = 3
k non empty subarrays
k2 k4 k6 concat in b
b += 0
cost(b) = min index i bi != i

3 2 
1 1 1

1 11
11 1

8 8
1 1 2 2 3 3 4 4
1 
1 , 1, 2, 2, 3, 3, 4, 4, 
1 2 3 4, 0
5 4
1 1 1 2 2
1 1, 1, 2,  2
1 1 2
5 4
1 1 10k 2 2
1 1, 10k , 2, 2,
10k 

- X1 X2 X3.. - Y1 Y2.. - Z1.. - 0
b = index(last)
b - k = space

3 2
5 1 1

if a[0] != 1: 
    return 1
1 2 3 4 5 6 7..
b[1] != 2
a1 a2 a3 a4 a5 
a1 [a2, a3, .., aj] aj+1 [aj + 2, .. ] ..
[a2 a3 aj aj + 2]
1 2 3 4 5
try to make a2 != 1
try to make a3 != 2
5 4 
1 1 1 2 2
5 3
1 1 1 2 2
[1 1 1] [2] [2]
2 0
10 1 2 3

5 4
1 1 2 10k 3
1, 1 2, 10k, 3
1 1, 2, 10k, 3

1 [1 10k] 2 [2]
> [1 1] 10k [2] 2

[1 1] 1 [2] 2
> 1 [1 1] 2 [2]

b[0] = k[1][0]
k[1][0] != 1

5 4
1 1 1 2 2

1 [1 1]
1 1 [1] 2 [2]
1 [1] 1 [2 2]

10 4
1 1 1 1 1 1 2 2 2 2
k[1][0] != 1
[1 1 1 1 1 1] [2] [2] [2 2]

1 2 2 2 2 1 1 1 1 1
1 [2 2]

[1 1 1]
what if I don't take it cause that's optimal at k+1?

5 3
1 1 1 2 2
[1 1 1] [1] [2] [2]
2 2 1 1 1
coming up with optimal strategy is too hard
I don't need to
I don't need to return number of subarrays I just need to return minimum b

8 4
1 1 2 2 3 3 4 4
8 4 
1 1 1 1 1 2 2 2

x x
1 1 2 2 3 3 4 4 5 5 
10 1 20 2 30 3










1 2 2 3

5 4
1 1 10k 2 2


"""

def main():
    t = int(stdin.readline())
    for i in range(t):
        n, k = map(int, stdin.readline().split())
        a = map(int, stdin.readline().split())

    # solve here
    print(n)
    print(s)
    print(x[0])
    print(y[n - 1])


def solve(n):
    return 0


main()
