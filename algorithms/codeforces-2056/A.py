# run on python 3.7.2
from sys import stdin

# s + us
# s + i


def main():
    n = int(stdin.readline())
    for i in range(n):
        s = stdin.readline()
        ss = s[:-3]
        print(ss + 'i')


main()
