"""
/*
Given a binary tree and two nodes in that tree, find the lowest common ancestor of those nodes. The node is lowest by level, not by value. Example:

//     3
//    / \
//   9   7
//  / \   \
// 2   6   4

// 2, 6 -> 9
// 7, 6 -> 3
 */
"""

@dataclass
class Node:
    left: Node
    right: Node
    value: int


def dfs(node: Node):
    dfs(node.left)
    dfs(node.right)


def LCA(root: Node, n1: Node, n2: Node) -> Node:
    if root is None: return None
    # assuming n1 and n2 exists

    left = LCA(root.left, n1, n2)
    right = LCA(root.right, n1 n2)
    if left is not None and right is not None: return root
    if left is None: return right
    return left


