# https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/ 
# ACCEPTED O(NLOGN)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def visit(node, x, y, d):
    d[x].append((y, node.val))
    if node.left:
        visit(node.left, x-1, y-1, d)
    if node.right:
        visit(node.right, x+1, y-1, d)


from collections import defaultdict


class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:

        d = defaultdict(list)
        visit(root, 0, 0, d)
        res = []
        for k in sorted(d.keys()):
            y_v = d[k]
            print(y_v)
            sorted_y_v = sorted(y_v, key=lambda x: (-x[0], x[1]))
            print(sorted_y_v)
            res.append([yv[1] for yv in sorted_y_v])
        return res

        
        
