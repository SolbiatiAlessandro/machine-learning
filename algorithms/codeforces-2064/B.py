# run on python 3.7.2
from sys import stdin

# 2 2 1 3 2
# 3 4

# 2 2 1 3 3 2
# 3 3

# 2 2 1 3 3 1 2
# 0

# 2 2 5 6 2 2 7 2
# 7

# 2 2 1 3
# 3 4

# 1 3 2 2
# 1 2

# 1 4 3 2 2
# 1 3

# 2 2 1 5 3
# 3 5



def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        print(solve(n, a))

def solve(n, a):

    fcount = {}
    for x in a:
        if x not in fcount.keys():
            fcount[x] = 1
        else:
            fcount[x] += 1

    max_len = 0
    max_l, max_r = -1, -1
    ix = 0

    while ix < len(a):
        x = a[ix]

        curr_len = 0
        l, r = -1, -1
        if fcount[x] == 1:
            l, r = ix + 1, ix + 1 # 2 2
            curr_len = 1
            ix += 1
            while ix < len(a) and fcount[a[ix]] == 1:
                r += 1
                curr_len += 1
                ix += 1
            if curr_len > max_len:
                max_len = curr_len
                max_l, max_r = l, r
            # print(ix, min_len)
        ix += 1

    if max_l == -1:
        return 0
    return f"{max_l} {max_r}"


main()
