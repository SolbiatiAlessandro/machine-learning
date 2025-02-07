# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
# accepted, beat 62.22%

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def traverse(node, p, q): 
    # return result, matches
    matches = []

    if node.left:
        res, left_matches = traverse(node.left, p, q) 
        if res: return res, None
        if left_matches: matches += left_matches
    
    if node.right:
        res, right_matches = traverse(node.right, p, q) 
        if res: return res, None
        if right_matches: matches += right_matches

    if node.val in [p, q]: matches.append(node.val)

    #print(node.val, matches, p, q)
    if len(matches) == 2: return node, None
    return None, matches


class Solution:

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # post order traversal
        # traverse left
        # traverse right
        # if p in left and q in right, res = 
        #return root.dfs(p, q)
        res, _ = traverse(root, p.val, q.val)
        return res

        
