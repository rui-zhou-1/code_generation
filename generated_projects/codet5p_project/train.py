# Implement next permutation, which rearranges numbers into the next greater permutation of numbers.

def next_permutation(arr, start, end):
    p = start
    i = start
    j = end

    while i < j:
        if arr[i] < arr[j]:
            break

        i += 1

    if i == j:
        print(arr)
        return

    k = i

    while k < j:
        if arr[k] > arr[j]:
            break

        k += 1

    arr[i], arr[j] = arr[j], arr[i]

    if k < end:
        while i < k:
            arr[i], arr[k] = arr[k], arr[i]
            i += 1
            k -= 1
        return

    if i < end:
        while i < end:
            arr[i], arr[end] = arr[end], arr[i]
            i += 1
            end -= 1
        return


if __name__ == '__main__':
    arr = [1, 4, 3, 2]
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len(arr) - 1)
    print(arr)
    next_permutation(arr, 0, len

from model import *
from dataset import *
