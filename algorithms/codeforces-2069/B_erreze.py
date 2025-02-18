# import sys
# input = sys.stdin.readline
def main():
    t = int(input())
 
    for _ in range(t):
        n, m = map(int, input().split())
        ar = []
        ac = {}
        acl = {}
        shared = {}
        
        for i in range(n):
            row = list(map(int, input().split()))
            ar.append(row)
            for j, num in enumerate(row):
                shared[num] = 0 if shared.get(num, 0) == 0 else 1
                l = acl.get(num, set())
                l.add((i,j)) 
                if (i,j-1) in l:
                    shared[num] = 1
                if (i-1,j) in l:
                    shared[num] = 1
                acl[num] = l
            
        # def shareSide(list: list):
        #     pos_set = set((x, y) for x, y in list)
        #     for x, y in list:
        #         if (x, y+1) in pos_set:
        #             return True
        #         if (x+1, y) in pos_set:
        #             return True
        #     return False
        
        # for i in range(n):
        #     for j in range(m):
        #         l = acl.get(ar[i][j], list())
        #         l.append((i,j)) 
        #         acl[ar[i][j]] = l
        # print(acl)
        # for i in acl:
        #     if shareSide(acl[i]):
        #         shared[i] = 1
        #     else:
        #         shared[i] = 0
                
        ans = 0
        # print(acl,shared)
        # for i 
        # ans = float('inf')
        l = []
        for i in shared:
            if shared[i] > 0 :
                l.append(2)
            else:
                l.append(1)
        l.sort()
        # print(l)
        for i in l:
            ans += i
        ans-=l[-1]
 
        print(ans)
 
 
# runner.py
import sys
from io import StringIO
 
def run_with_file(input_file='./latest/b.txt'):
    # Save the original stdin
    original_stdin = sys.stdin
    
    # Replace sys.stdin with file contents
    with open(input_file, 'r') as f:
 
        sys.stdin = StringIO(f.read())
        
    try:
        # Run the main solution
        main()
    finally:
        # Restore original stdin
        sys.stdin = original_stdin
 
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_with_file()
    else:
        main()  # Regular execution for submission
