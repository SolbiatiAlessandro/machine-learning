"""
"quick brown fox jumped over" label

N = 2

"quick brown" label
"brown fox" label
"fox jumped" label
"jumped over" label

n-gram

Input:

L = [a, b, c, d, e, f]
N = 3

Output:

[
  [a, b, c],
  [b, c, d],
  [d, e, f],
  [c, d, e],
]

def ngram(words, n):
	res = []
	for i, w in enumerate(words):
		if i+n == len(words):
			break
		res.append(words[i: i+n])
	return res

words = ["a", "b", "c", "d", "e", "f"]
n = 3

print(ngram(words, n))

words = ["a", "b", "c", "d", "e", "f"]
n = 6

print(ngram(words, n))

words = ["a", "b"]
n = 1

print(ngram(words, n))
"""
		
# we introduce skip budget

# Example input: L = [a, b, c, d, e], N = 3, K = 2

# Output: [a, b, c], [a, b, d], [a, b, e], [a, c, d], [a, c, e], [a, d, e], [b, c, d], [b, c, e], [b, d, e], [c, d, e]

a = ['a', 'b', 'c', 'd', 'e']




def ngram(a, n, k):

    def f(i, n, k, a):
        # n: letters left
        # k: skips left
        # i current index
        if n == 0:
            res = [a[i]]
        else:
            res = []
            for j in range(1, min(k+2, len(a) - i)):
                children = f(i + j, n - 1, k - (j - 1), a)
                for c in children:
                    res.append(a[i] + c)
        if cache[i][n][k - 1] == 0: 
            cache[i][n][k - 1] = res
        else:
            print("hitting cache")
        return res


    cache = [[[0] * k] * n] * len(a)
    res = []
    for i in range(len(a)):
        _ngram = f(0, n - 1, k, a[i:])
        if _ngram:
            res += _ngram
    return res, cache

print(ngram(a, 3, 2))


