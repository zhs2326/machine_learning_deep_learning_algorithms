import collections
import numpy as np


def k_means(input_data, k, max_iter=3000, metric='l2', same_iter=10):
    k_centers = input_data[np.random.choice(input_data.shape[0], k, replace=False), :]
    prev_arrangement = arrange(input_data, k_centers, metric)
    same_it = 0
    it = 0

    while it < max_iter:
        # handle empty center situation
        try:
            k_centers = find_centers(input_data, k_centers, prev_arrangement)
        except ZeroDivisionError:
            k_centers = input_data[np.random.choice(input_data.shape[0], k, replace=False), :]
            prev_arrangement = arrange(input_data, k_centers, metric)
            same_it = 0
            it = 0
            continue

        curr_arrangement = arrange(input_data, k_centers, metric)
        if curr_arrangement == prev_arrangement:
            same_it += 1
            if same_it == same_iter:
                return k_centers, curr_arrangement
        else:
            same_it = 0
        prev_arrangement = curr_arrangement

    return k_centers, curr_arrangement


def arrange(input_data, k_centers, metric='l2'):
    arrangement = collections.defaultdict(int)
    for i, row in enumerate(input_data):
        if metric == 'l1':
            nearest_center = np.argmin([np.linalg.norm(row-center) for center in k_centers])
        if metric == 'l2':
            nearest_center = np.argmin([np.linalg.norm(row-center, ord=1) for center in k_centers])

        arrangement[i] = nearest_center

    return arrangement


def find_centers(input_data, prev_centers, arrangement):
    dim = input_data.shape[1]
    k = prev_centers.shape[0]
    new_centers = np.repeat(np.array([np.zeros(dim+1)]), k, axis=0)

    for i, row in enumerate(input_data):
        new_centers[arrangement[i]][:-1] = np.add(new_centers[arrangement[i]][:-1], row)
        new_centers[arrangement[i]][-1] += 1

    for i in range(new_centers.shape[0]):
        if new_centers[i][-1] == 0:
            raise ZeroDivisionError
        else:
            new_centers[i][:-1] = np.divide(new_centers[i][:-1], new_centers[i][-1])

    new_centers = new_centers[:, :-1]

    return new_centers


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    test_input_data = np.random.rand(50, 2)
    K = 4
    centers, arr = k_means(test_input_data, K, max_iter=5000, metric='l2', same_iter=10)

    colors = ['red', 'green', 'blue', 'orange']
    for i, row in enumerate(test_input_data):
        x1, x2 = row
        plt.scatter(x1, x2, color=colors[arr[i]])

    x1, x2 = centers.T
    plt.scatter(x1, x2, color='black', s=100)
    plt.show()
