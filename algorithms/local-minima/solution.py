"""
Given an array of integers, find any one local minimum from the array. A local minimum is defined as an integer in the array that is less than or equal to its neighbors.

[5, 9, 7, 10, 12] => 5 or 7

#q1 edge? counts
#q2 null

solution: binary search

forgot asking about equal

check empty array
"""


def local_minima(a):
    s, e = 0, len(a) - 1

    while s < e:
        m  = s + ((s - e) / 2)
        val = a[m]
        if val <= array[m + 1]):
            # move left
            e = m
        else vall > array[m + 1]:
            s = m + 1
    return a[s]


"""
follow up, what if is stricly less

what do you do if is equal?

O(n)


"""



