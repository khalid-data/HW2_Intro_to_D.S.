# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import data
import clustering
import sys


def main(argv):
    df = data.load_data(argv[1])
    list = ['t1', 't2']
    clustering.transform_data(df, list)

if __name__ == '__main__':
    main(sys.argv)

