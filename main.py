# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data import *
import data
import clustering
from clustering import *
import sys


def main(argv):
    df = data.load_data(argv[1])
    fetures = ['cnt', 'hum']
    x = transform_data(df, fetures)


if __name__ == '__main__':
    main(sys.argv)

