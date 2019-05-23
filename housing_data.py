"""Utility functions for loading the housing data set."""
import pandas as pd


def format_date_columns(dataframe: pd.DataFrame, columns, format='%Y/%m/%d'):
    """Format date columns to datetime.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing the date columns.
    columns : list
        List of date column names.
    format : str
        Format used to transform the date. Default to '%Y/%m/%d'.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame.

    """
    df_copy = dataframe.copy()
    for date_column in columns:
        df_copy[date_column] = pd.to_datetime(
            df_copy[date_column],
            format=format)

    return df_copy


def get_prefixed_column_names(dataframe: pd.DataFrame, prefix):
    """Return list of columns in dataframe with given prefix.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe containing the date columns.
    prefix : str
        Description of parameter `prefix`.

    Returns
    -------
    list
        List of columns with prefix.

    """
    columns = []

    for column_name in dataframe.columns:
        if column_name.startswith(prefix):
            columns.append(column_name)

    return columns


import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print('Distance: ' + repr(distance))
