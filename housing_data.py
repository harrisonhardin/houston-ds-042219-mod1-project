"""Utility functions for loading the housing data set."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import cos, asin, sqrt
import time

DATA_CSV_PATH = 'kc_house_data.csv'  # CSV location.


def load_housing_data(with_cat_columns=False):
    """Load King County housing data csv.

    Parameters
    ----------
    with_cat_columns : boolean
        Include in the DataFrame output the categorical variables.
        Default False.

    Returns
    -------
    pd.DataFrame
        The corresponding housing cleaned dataset.

    """
    # Load corresponding DataFrame
    data = pd.read_csv('kc_house_data.csv')

    # Perfom cleaning on empty/NaN variables

    # Yr renovated : Fill 0 Year renovated with Yr Built
    data['yr_renovated'] = data['yr_renovated'].fillna(data['yr_built'])
    data.loc[data['yr_renovated'] == 0, 'yr_renovated'] = \
        data.loc[data['yr_renovated'] == 0, 'yr_built']

    # Sqft Basement : Remove non numerical values and set them to null
    data.loc[data['sqft_basement'] == '?', 'sqft_basement'] = 0

    # View : Set missing values to null
    data['view'].fillna(0, inplace=True)

    # Waterfront : Set missing values to null
    data['waterfront'].fillna(0, inplace=True)

    # Transforming date, yr_built, yr_renovated  to datetime
    data = format_date_columns(data, ['date'], format='%m/%d/%Y')
    date_columns = ['yr_built', 'yr_renovated']
    data = format_date_columns(data, date_columns, format='%Y')

    # Waterfront : Transforming waterfront to boolean
    data['waterfront'] = data['waterfront'].astype(bool)

    # Transform  Categorical Variables using one-hot-encoding
    if with_cat_columns:
        categorical_variables = ['waterfront', 'condition', 'grade',
                                 'bedrooms', 'bathrooms', 'floors',
                                 'view', 'zipcode']

        # Build dummy data series
        water_dummies = pd.get_dummies(data['waterfront'], prefix="water")
        cond_dummies = pd.get_dummies(data['condition'], prefix="cond")
        grade_dummies = pd.get_dummies(data['grade'], prefix="grade")
        bed_dummies = pd.get_dummies(data['bedrooms'], prefix='bed')
        bath_dummies = pd.get_dummies(data['bathrooms'], prefix='bath')
        floor_dummies = pd.get_dummies(data['floors'], prefix='floor')
        view_dummies = pd.get_dummies(data['view'], prefix='view')
        zipcode_dummies = pd.get_dummies(data['zipcode'], prefix='zip')

        # Drop and Concat categorical variables
        data = data.drop(categorical_variables, axis=1)
        data = pd.concat([
                            data,
                            water_dummies,
                            cond_dummies,
                            grade_dummies,
                            bed_dummies,
                            bath_dummies,
                            floor_dummies,
                            view_dummies,
                            zipcode_dummies
                        ], axis=1)

    return data


def convert_categorical_variables(dataframe: pd.DataFrame,
                                  categorical_variables: [], dummy=True):
    """Short summary.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Description of parameter `dataframe`.
    categorical_variables : []
        Description of parameter `categorical_variables`.
    dummy : boolean
        one-hot encoding or category
    Returns
    -------
    type
        Description of returned object.

    """
    df_copy = dataframe.copy()

    # Transform  Categorical Variables using one-hot-encoding or category
    for variable in categorical_variables:
        if dummy:
            var_dummies = pd.get_dummies(df_copy[variable], prefix=variable[:3])
            df_copy.drop(variable, axis=1, inplace=True)
            df_copy = pd.concat([df_copy, var_dummies], axis=1)
        else:
            df_copy[variable] = df_copy[variable].astype('category')

    return df_copy


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


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between 2 points defined by lat/lon.

    The calculation is done based on Haversine formula.
    see https://en.wikipedia.org/wiki/Haversine_formula for details.

    Parameters
    ----------
    lat1 : float
        Latitude of first point.
    lon1 : float
        Longitude of first point.
    lat2 : float
        Latitude of second point.
    lon2 : float
        Longitude of second point.

    Returns
    -------
    float
        Distance between the two coordinates in miles.

    """
    miles_constant = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    mi = miles_constant * c
    return mi


def get_closest_properties(
        dataframe: pd.DataFrame,
        property: pd.DataFrame,
        radius: float):
    """Get the closest properties.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Properties dataset to get the closest properties.
    property : pd.DataFrame
        Property to get the closest locations.
    radius : float
        Radius in miles to look for properties.

    Returns
    -------
    pd.DataFrame
        List of the closest properties within radius.

    """
    # Make sure to remove the actual property from the list.
    return dataframe[dataframe.apply(
        lambda row: calculate_distance(property['lat'], property['long'], row['lat'], row['long']) < radius and row['id'] != property['id'], axis=1)]


def get_price_per_sqft_living(dataframe: pd.DataFrame):
    """Return list of price per sqft dataset from closest neighbors.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original properties dataset to add the price per sqft.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the price per sqft from closest neighbors.

    """
    # Timer init
    start_time = time.time()

    price_per_sqft = pd.Series([])
    for idx, property_ds in dataframe.iterrows():
        radius = 1  # Set radius 1 mile around location

        # Get nearest locations
        closest_properties_df = get_closest_properties(
            dataframe, property_ds, radius)

        # Increment search area by 1 mile around location if empty.
        while closest_properties_df.empty:
            radius += 1
            closest_properties_df = get_closest_properties(
                dataframe, property_ds, radius)

        price_per_sqft[idx] = average_price_per_sqft_living(
            closest_properties_df)

    # Timer output
    print("--- %s seconds ---" % (time.time() - start_time))

    return price_per_sqft


def average_price_per_sqft_living(dataframe: pd.DataFrame):
    """Calculate the average price per sqft in properties dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Properties dataset to get the average price per sqft.

    Returns
    -------
    float
        Average properties price per sqft.

    """
    return dataframe['price'].mean()/dataframe['sqft_living'].mean()


def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out=0.05,
                       verbose=True):
    """Perform a forward-backward feature selection based on p-value.

    Perform a forward-backward feature selection based on p-value
    from statsmodels.api.OLS

    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details.

    Parameters
    ----------

        X : pandas.DataFrame
            pandas.DataFrame with candidate features
        y : list
            like with the target
        initial_list : list
            List of features to start with (column names of X)
        threshold_in : float
            Include a feature if its p-value < threshold_in
        threshold_out : float
            Exclude a feature if its p-value > threshold_out
        verbose : boolean
            Whether to print the sequence of inclusions and exclusions

    Returns
    -------
    list
        List of selected features

    """
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y,
                           sm.add_constant(
                               pd.DataFrame(X[included+[new_column]]))
                           ).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
