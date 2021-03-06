import pandas as pd
import datetime
from heapq import nlargest, nsmallest
from datetime import datetime


def load_data(path):
    """
    reads and returns the pandas DataFrame
    :param path: path to csv file
    :return: the data frame
    """
    df = pd.read_csv(path)
    return df


def add_new_columns(df):
    """
    adds columns to df and returns the new df
    :param df: data frame
    :return: new dataframe
    """

    df['season_name'] = df['season'].apply(lambda season: "spring" if season == 0 else "summer" if season == 1
    else "fall" if season == 2 else "winter")
    df['Hour'] = df['timestamp'].apply(return_hour)
    df['Day'] = df['timestamp'].apply(return_day)
    df['Month'] = df['timestamp'].apply(return_month)
    df['Year'] = df['timestamp'].apply(return_year)
    df['is_weekend_holiday'] = df.apply(lambda x: 0 if (x.is_holiday == 0 and x.is_weekend == 0)
    else 1 if (x.is_holiday == 0 and x.is_weekend == 1)
    else 2 if (x.is_holiday == 1 and x.is_weekend == 0) else 3, axis=1)
    df['t_diff'] = df.apply(lambda x: x.t2 - x.t1, axis=1)
    return df


def data_analysis(df):
    """
    prints statistics on the transformed df
    :param df: given data frame
    :return: none
    """
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()
    dict = {}
    keys = corr.keys()
    for i in keys:
        for j in keys:
            if i == j or (i, j) in dict.keys() or (j, i) in dict.keys():
                continue
            dict[(i, j)] = (abs(corr[i][j]))

    largest_five = nlargest(5, dict, key=dict.get)
    smallest_five = nsmallest(5, dict, key=dict.get)

    print("Highest correlated are: ")
    i = 1
    for key in largest_five:
        print(str(i) + '. ' + str(key) + ' with ' + "{:.6f}".format(dict[key]))
        i = i + 1

    print()
    print("Lowest correlated are: ")
    """ 6 figures after decimal point"""
    i = 1
    for key in smallest_five:
        print(str(i) + '. ' + str(key) + ' with ' + "{:.6f}".format(dict[key]))
        i = i + 1

    season_mean = round(df.groupby(['season_name']).mean(), 2)
    print()

    season_mean.apply(lambda x: print(str(x.name) + ' average t_diff is ' + "{:.2f}".format(x.t_diff)), axis=1)
    print('All average t_diff is ' + "{:.2f}".format((df[['t_diff']].mean()['t_diff'])))


def return_hour(str):
    """
    returns the hour part of the str
    :param str: string of the date
    :return: hour
    """
    datetime_object = datetime.strptime(str, '%d/%m/%Y %H:%M')
    return datetime_object.hour


def return_year(str):
    """
    returns the year part of the str
    :param str: string of the date
    :return: year
    """
    datetime_object = datetime.strptime(str, '%d/%m/%Y %H:%M')
    return datetime_object.year


def return_month(str):
    """
    returns the month part of the str
    :param str: string of the date
    :return: month
    """
    datetime_object = datetime.strptime(str, '%d/%m/%Y %H:%M')
    return datetime_object.month


def return_day(str):
    """
    returns the day part of the str
    :param str: string of the date
    :return: day
    """
    datetime_object = datetime.strptime(str, '%d/%m/%Y %H:%M')
    return datetime_object.day
