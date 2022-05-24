# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data import *
import data
import clustering
from clustering import *
import sys


def main(argv):
    print('Part A: ')
    df = data.load_data(argv[1])
    df = add_new_columns(df)
    data_analysis(df)
    fetures = ['cnt', 'hum']
    x = transform_data(df, fetures)

    print()
    print('Part B: ')

    k_values = [2, 3, 5]
    for k in k_values:
        labels, centroids = kmeans(x, k)
        if k != 2:
            print()
        print('k = ' + str(k))
        visualize_results(x, labels, centroids, f'plot{k}.png')


if __name__ == '__main__':
    main(sys.argv)
