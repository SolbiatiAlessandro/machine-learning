# run on python 3.7.2
from sys import stdin


def main(debug=False):
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        #s = stdin.readline()
        a = list(map(int, stdin.readline().split()))
        a = [0] + a + [0]
        #if debug: print(a, n)
        print(solve(a))

def solve(a):
    n = len(a)
    print(a, n)
    if len(a) <= 1: return "YES"
    ix = 1
    #print(ix)
    while ix < n:
        if a[ix] == 1:
            if a[ix - 1] == 1: # previous was 1
                return "NO"
            while a[ix] == 1: # previous was 0
                ix += 1
            return solve(a[ix + 1:])
        ix += 1
    return "YES" # hack all 0s


    


main()
