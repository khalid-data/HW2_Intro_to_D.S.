import numpy as np
import pandas as pd
from math import pow
import pylab as plb
import matplotlib.pyplot as plt

np.random.seed(2)


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


# ====================
def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """

    array1 = df[features[0]].to_numpy()
    array2 = df[features[1]].to_numpy()
    sum1 = np.sum(array1)
    sum2 = np.sum(array2)
    min1 = np.min(array1)
    min2 = np.min(array2)
    for i in range(array1.size):
        array1[i] = (array1[i] - min1) / sum1

    for j in range(array2.size):
        array2[j] = (array2[j] - min2) / sum2

    transformed_data = np.vstack((array1, array2))
    transformed_data = add_noise(transformed_data.transpose())  # change shape from (n,2) to (2,n)
    return transformed_data  # get back shape


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """
    curr_centroids = choose_initial_centroids(data, k)
    prev_centroids = np.zeros(shape=(k, 2))
    while not np.array_equal(curr_centroids, prev_centroids):
        labels = assign_to_clusters(data, curr_centroids)
        prev_centroids = curr_centroids
        curr_centroids = recompute_centroids(data, labels, k)
    return labels, curr_centroids


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    new_data = np.zeros(shape=(len(data), 3))
    for i in range(len(data)):
        new_data[i][0] = data[i][0]
        new_data[i][1] = data[i][1]
        new_data[i][2] = labels[i]

    plt.xlabel('cnt')
    plt.ylabel('hum')
    # Alternatively:

    plt.scatter(new_data[:, 0], new_data[:, 1], c=new_data[:, 2])
    plt.title('Result for kmean with k = ' + str(len(centroids)))
    plt.show()
    # plt.savefig(path)




def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    g = sum([pow((i - j), 2) for i, j in zip(x, y)])
    return g ** (1 / 2)
    # return distance


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    for each point in data find_closest_centroid and assign it to that

    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """
    labels = np.zeros(shape=data.size)##lennnnnnnnnnnnnn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(len(data)):
        index = find_closest_centroid(centroids, data[i])
        labels[i] = index
    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    clusters = np.zeros(shape=(k, 2))
    for i in range(k):
        cnt = 0
        for j in range(len(data)):
            if labels[j] == i:
                clusters[i][0] += data[j][0]
                clusters[i][1] += data[j][1]
                cnt += 1
        clusters[i][0] = clusters[i][0] / cnt
        clusters[i][1] = clusters[i][1] / cnt
    return clusters


def euclidian_dist(y, x):
    """
    Euclidean distance between vectors x, y 2 dimintional
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    g = sum([pow((i - j), 2) for i, j in zip(y, x)])
    return g ** (1 / 2)


def find_closest_centroid(array, x):
    i = 0
    min_index = 0
    min_dist = euclidian_dist(array[i], x)

    for i in range(len(array)):
        temp_dist = euclidian_dist(array[i], x)
        if temp_dist < min_dist:
            min_dist = temp_dist
            min_index = i
    return min_index
