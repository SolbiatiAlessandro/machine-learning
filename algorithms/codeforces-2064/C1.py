# run on python 3.7.2
from sys import stdin



def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        print(f"input#{i}: {a}")
        print(solve(n, a))

def solve(n, a, debug=True):
    if n == 0: return 0

    right_sum, right_ix = 0, len(a) - 1
    while right_ix >= 0 and a[right_ix] >= 0:
        right_sum += a[right_ix]
        right_ix -= 1


    left_sum, left_ix = 0, 0
    while left_ix < (len(a)) and a[left_ix] < 0:
        left_sum += -1 *  a[left_ix]
        left_ix += 1


    res = 0
    if debug: print(left_ix, right_ix)
    #if debug: print(left_total, right_total)
    if right_sum > left_sum:
        res += abs(a[right_ix - 1])
        right_ix = right_ix - 1
        left_ix = 0
    else:
        res += abs(a[left_ix +1])
        left_ix = left_ix + 1
        right_ix = len(a)

    new_a = a[left_ix:right_ix+1]
    if debug: print(f"res: {res}")
    if debug: print(left_ix, right_ix)
    return res + solve(len(new_a), new_a)
    #return res

main()
