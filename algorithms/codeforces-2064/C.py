# run on python 3.7.2
from sys import stdin



def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        # print(f"input#{i}: {a}")
        print(solve(n, a))

def solve(n, a, debug=False):
    if n == 0: return 0
    if debug: print(0, len(a))

    res = 0

    # cut 
    left_sum, left_ix = 0, 0
    while left_ix < len(a) and a[left_ix] >= 0:
        left_sum += a[left_ix]
        left_ix += 1

    right_sum, right_ix = 0, len(a) - 1
    while right_ix >= 0 and a[right_ix] < 0:
        right_sum += -1 * a[right_ix]
        right_ix -= 1

    res += left_sum + right_sum

    # evaluate

    a = a[left_ix:right_ix + 1]
    if len(a) == 0: return res

    if debug: print(left_ix, right_ix + 1)
    right_sum, right_ix = 0, len(a) - 1
    while right_ix >= 0 and a[right_ix] >= 0:
        right_sum += a[right_ix]
        right_ix -= 1
    left_sum, left_ix = 0, 0
    while left_ix < len(a) and a[left_ix] < 0:
        left_sum += -1 *  a[left_ix]
        left_ix += 1
    if debug: print("sum:", right_sum, left_sum)
    if debug: print("ix:", right_ix, left_ix)

    if right_sum > left_sum:
        res += abs(a[left_ix])
        left_ix = left_ix + 1
        right_ix = len(a) - 1
        #res += right_sum
    else:
        res += abs(a[right_ix])
        right_ix = right_ix - 1
        left_ix = 0
        #res += left_sum
    a = a[left_ix:right_ix + 1]
    if debug: print(left_ix, right_ix + 1)

    return res + solve(len(a), a)
    #return res

main()
