# run on python 3.7.2
from sys import stdin
"""

n, 

0: nothihng
X: B1: a1, a2, -> B2
Y: B1: x, B2: y  -> if x == y: x+=1

B1 == B2 

6
3 3 4 5 3 3
0,0,0,0,0,0

must be even
split in two 
for each non identical character has to increase
N^2 I can increase them all

1 1 4 4
14 14
3 4 33
3 3 3 4

N^2 * logn

3 3 4 5 3 3

f[2] = 2
f[4] = 4

2
3
1
2

7 -> 4 -> 3 -> 1 -> 2
0 -> 3 -> 1 -> 2 -> 1
0 -> 2 - > 1 -> 2

1 -> 2

7 - 3 = 4
1
1
2

3 -> 3 - x
0
3 -> 3 + 1
x = 1 + # of zeroes between

4 -> 4 - 1 = 3
1
1 -> 1 + 1

6
1
6 -> 5 --> 4
1 --> 2
1 - > 2

a is the frequency map
ix = len(a) - 1 #start at the bottom
while True:
    ix == 0? solved print yes
    is a[ix] it odd:

    # if is it odd continue
    iy = ix # go up one by one
    zero_counts = 0
    for iy -= 1
        if a[iy] == 0:
            a[iy] = 1
            zero_counts += 1
        if value of a[iy] >= 2
            move value down 
            a[iy] -= 1 + zero_counts
            a[ix] += 1 # it's even
            ix -= 1
            break
    if a[ix] % 2 == 1:
        return print("No")



4 - 1= 3
1
1

2
8

f[3] = 4
f[4] = 1
f[5] = 1

f[9] = 2
f[10] = 8

f[1] = 2
2, 2
3, 2
f[4] = 2


3 3 3, 3 4 5
3 3 3, 3 4 5

frequency map of right f[3] = 1


for i in left:
    balanced?
    if no, can I increase?

    for j in right:
        incraase i by one
        check
            



3 3
4 5 3 3

3 3 -> 4, 3
3 3 3 -> 5, 3

def x(a, m, i):
    rec = x(a, m[i] = 1, i+1)
    return [rec+a[i]]

0,0,0,0,0,1

21 - 3, 0 + 3
0,0,0,1,0,1
21 - 5, 0 + 5

21 - x, 0 + x
21 - x == 0 + x
x = 21/2 = 10.5

2 1
0,0
1 1

2 1

3, 3 4 5 3 3
4
5

3 = 3
3+3 = 6
3+0 = 3
3+3+5 = 11
3+0+5 = 8
2 ^ n 

3 3 4 5
3, 3 4 5
6, 3 4 5

2 2 2 4 4 4
2 2 4, 4 4 2
3 3 4, 4 4 2

2 | 2, 3 | 2
2 | 




2 2 2 4 4 4
2 2 4, 2 4 4
3 2 4, 2 4 4
3 3 4, 2 4 4


numbers can be moved freely from b1 to b2
1 1 0
1 1

10 -> 6
1
1
3 -> 1
1
1
1
6

10 -> 9
0
0
3 -> 2
0
0
0
6


"""



def main():
    t = int(stdin.readline())
    for i in range(t):
        n = int(stdin.readline())
        a = list(map(int, stdin.readline().split()))
        solve(a, n, debug=False)

def solve(a, n, debug=False):

    if debug: print(list(a), n)
    frequency_map = [0] * (n + 1)
    for x in a:
        frequency_map[x] += 1

    ix = len(frequency_map) - 1 #start at the bottom
    if debug: print(f"frequency map at step {ix}: {frequency_map}")
    while ix > 0:
        even = frequency_map[ix] % 2 == 0
        if debug: print(f"round {ix}, {frequency_map[ix]} - at beginning even? {even}")
        if not even:
            iy = ix - 1 # go up one by one
            zeros_to_fill = 0
            #print(iy, zeros_to_fill)
            while iy >= 0 and not even:
                #print(iy, zeros_to_fill, frequency_map[iy])
                if frequency_map[iy] == 0:
                    frequency_map[iy] = 1
                    zeros_to_fill += 1
                if frequency_map[iy] >= 2:
                    max_can_move = frequency_map[iy] - 1
                    # what happen if I need to move from more numbers?
                    moving = min(zeros_to_fill, max_can_move)
                    #print(f"moving {max_can_move} {moving}")
                    frequency_map[iy] -= moving
                    zeros_to_fill -= moving
                    left_after_filling_zero = max_can_move - moving
                    #print(f"after {zeros_to_fill} {left_after_filling_zero}")
                    if left_after_filling_zero > 0:
                        frequency_map[ix] += 1 # it's even
                        even = True
                        frequency_map[iy] -= 1
                iy -= 1
        if debug: print(f"frequency map at step {ix}: {frequency_map}")
        if debug: print(f"round {ix}, {frequency_map[ix]} - at end even? {even}")
        #print(frequency_map[ix] % 2)
        if not even:
            print("No"); return # we were not able to get last one even
        ix -= 1
    print("Yes"); return # ix == they are all even


main()
