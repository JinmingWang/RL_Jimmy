import numpy as np

def multiArgmax(arr):
    max_val = np.max(arr)
    return np.where(arr == max_val)[0]

def multiArgmin(arr):
    min_val = np.min(arr)
    return np.where(arr == min_val)[0]

if __name__ == "__main__":
    arr = np.array([1, 2, 3, 2])
    print(multiArgmax(arr))
    arr = np.array([1, 5, 3, 2])
    print(multiArgmax(arr))
    arr = np.array([1, 1, 1, 1])
    print(multiArgmax(arr))
