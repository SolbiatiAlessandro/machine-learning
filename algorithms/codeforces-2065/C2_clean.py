# run on python 3.7.2j
import bisect
from sys import stdin

def main():
    t = int(stdin.readline())
    x = [None] * t
    y = [None] * t
    for i in range(t):
        n, m = map(int, stdin.readline().split())
        a  = list(map(int, stdin.readline().split()))
        b  = list(map(int, stdin.readline().split()))
        solve(n, m, a, b, _print=True)

def solve(n, m, a, b, _print=False, debug=False):
    b = sorted(b) #1, 6, 8
    maxes = [None] * n #[n,n,n,n]
    maxes[n - 1] = max(b[-1] - a[n - 1], a[n - 1]) #[n,n,2,5]
    i = n - 2 # 3
    while i >= 0: # 1
        previous_max = maxes[i + 1] # 2
        current_a = a[i] # 4

        search = previous_max + current_a # 2 + 4 = 6
        current_bi = bisect.bisect_right(b, search) # 1

        current_max = -10e10
        if current_bi > 0:
            best_b = b[current_bi - 1] # 4
            current_max = max(best_b - current_a, current_max)

        if current_a <= previous_max:
            current_max = max(current_a, current_max)

        if current_max == -10e10:
            print("NO")

        maxes[i] = current_max # [n,n,n,0,995]
        i -= 1
    print("YES")

main()
