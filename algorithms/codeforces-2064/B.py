# run on python 3.7.2
import sys

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
    input_data = sys.stdin.buffer.read().split()
    t = int(input_data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(input_data[index])
        index += 1
        a = list(map(int, input_data[index:index+n]))
        index += n
        results.append(solve(n, a))
    sys.stdout.write("\n".join(results))

from collections import Counter
def solve(n, a):

    fcount = Counter(a)

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
        return "0"
    return f"{max_l} {max_r}"


main()
