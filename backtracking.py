import time


def solve(items, bin_capacity):
    tic = time.time()
    bins = []
    items.sort(reverse=True)
    while True:
        if search(bins, items, bin_capacity):
            toc = time.time()
            return len(bins), bins, toc - tic
        bins.append([])


def search(bins, items, bin_capacity):
    if not items:
        return True
    if items[0] > bin_capacity:
        raise ValueError("Invalid item size provided!")
    best_bin = -1
    best_space = bin_capacity + 1
    for i in range(len(bins)):
        space = bin_capacity - sum(bins[i])
        if items[0] <= space < best_space:
            best_bin = i
            best_space = space
    if best_bin == -1:
        return False
    bins[best_bin].append(items[0])
    if search(bins, items[1:], bin_capacity):
        return True
    bins[best_bin].pop()
    return False


items = [8, 1, 4, 3, 7, 5, 2]
bin_capacity = 10
min_bins, bins, execution_time = solve(items, bin_capacity)
print(f"The minimum number of bins required is: {min_bins}")
print(f"The bins: {bins}")
print(f"Execution time: {execution_time} seconds")