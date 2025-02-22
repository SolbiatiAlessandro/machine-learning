from sys import stdin
 
# greedy
# pick always the largest set you can?
 
# need to pick the diagonals
# but how do I pick the target
 
# if a number share side with same number , number is in both set
# first set
# second set
# Q: are there only two sets? yes
 
 
set1 = [0] * 50000
set2 = [0] * 50000

t = int(stdin.readline())
for i in range(t):
    n, m = map(int, stdin.readline().split())
    a = [[-1 for _ in range(m + 2)]]
    for _ in range(n):
        row = [-1]
        row += list(map(int, stdin.readline().split())) 
        row.append(-1)
        a.append(row)
    a.append([-1 for _ in range(m + 2)])
    print(solve(a, n, m))
 
 
def has_neighbour(a, i, j):
    #print(f'has_neighbourh {a[i][j]} {i} {j}')
    for ii in [i-1, i+1]:
        nn = a[ii][j]
        #print(f'{nn}, a[{ii}, {jj}]')
        if nn != -1:
            if a[i][j] == nn:
                return True
    for jj in [j-1, j+1]:
        nn = a[i][jj]
        #print(f'{nn}, a[{ii}, {jj}]')
        if nn != -1:
            if a[i][j] == nn:
                return True
    return False
 
from collections import defaultdict
def solve(a, n, m, debug=True):
    #print(a)
    set1 = list(map(lambda x: 0 for x in set1))
    set2 = list(map(lambda x: 0 for x in set2))
   
    lenset1 = 0
    lenset2 = 0
 
    for i in range(n):
        for j in range(m):
            x = a[i+1][j+1]
            if set1[x] == 0:
                set1[x] = 1
                lenset1 += 1
            if has_neighbour(a, i+1, j+1):
                if set2[x] == 0:
                    set2[x] = 1
                    lenset2 += 1
            #print(f"{x} added to set1 {set1[x]} added to set {set2[x]}")
 
    res = -1 if lenset2 == 0 else -2
    res += lenset1 + lenset2
    return res
 
main()
