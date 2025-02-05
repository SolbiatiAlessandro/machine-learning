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
"""

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
		


