# run on python 3.7.2
from sys import stdin


def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = []
        for row_ix in range(n):
            row = list(map(int, stdin.readline().split()))
            a.append(row)
        print(solve(n, a))


def solve(n, a):
    suffix_len = [0] * n # row_id
    for ix in range(1, n+1):
        col_ix = n - ix
        for row_ix in range(n):
            if a[row_ix][col_ix] == 1 and suffix_len[row_ix] == ix-1:
                suffix_len[row_ix] += 1


    res = 1
    #print(sorted(suffix_len))

    for v in sorted(suffix_len):
        if v >= res:
            res += 1
    return min(res, n)



        #1
       #11
      #111

main()
