# %% [markdown]
# # Build Regression Model based on data_per_zipcode
#
# As seen during our collinearity analysis it seems that zipcode might affect
# the property price.

# %%
# Set auto reload
import pandas as pd
import numpy as np
import housing_data as hd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from statsmodels.formula.api import ols

%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data = hd.load_housing_data(with_cat_columns=False)
data.head()


# [markdown]
# **Evaluate dataset size per zipcode**

# %%
# grouped_data = data.groupby('zipcode')
# grouped_data.count()
# for name, group in data.groupby('zipcode'):
#     # print(name, ' contains ', len(group))
#     # print(type(group))
#     # plt.scatter(group['sqft_living'], group['price']);
#     # sns.scatterplot(x='sqft_living', y='price', hue='grade', data=group)
#     g = sns.FacetGrid(group, col="grade")
#     g.map(plt.scatter, "price", "sqft_living")
#     g.add_legend()
#     # plt.show()

# %%
g = sns.factorplot("zipcode", data=data, aspect=1.5, kind="count", color="b")
g.set_xticklabels(rotation=30)


# %%
# Split dataset in two group of zipcode for display purpose.
zipcodes = data['zipcode'].unique()
zipcodes.sort()
zipcodes_median = zipcodes[(len(zipcodes) // 2): (len(zipcodes) // 2) + 1][0]

print(zipcodes_median)

data_set1 = data.loc[data['zipcode'] <= zipcodes_median]
data_set2 = data.loc[data['zipcode'] > zipcodes_median]

# %%
plt.figure(figsize=(10,10))
g = sns.FacetGrid(data_set2, col="zipcode", hue='condition', col_wrap=3)
g.map(plt.scatter, "sqft_living", "price")
g.set(xlim=(0, 6000), ylim=(0, 3000000))
g.set_xlabels('')
g.set_ylabels('Property price')
# g.add_legend();


# %%

sns.set(style="ticks", palette="pastel")

# Load the example tips dataset
# tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="zipcode", y="price", palette=["m", "g"], data=data_set2)
plt.ylim(0, 4000000)
sns.despine(offset=10, trim=True)

# for name, group in data.groupby('zipcode'):
#     g = sns.boxplot(x="zipcode", y="price", palette=["m", "g"], data=group)
#     sns.despine(offset=10, trim=True)

# %%
toto = input('Enter zipcode')
print(toto)

# %%
print(data.columns)

# %%
# from statsmodels.formula.api import ols
# cleaned_data = data.drop(columns=['id', 'date', 'bedrooms', 'bathrooms',
#                                   'floors', 'waterfront', 'view',
#                                   'sqft_basement', 'yr_built', 'yr_renovated',
#                                   'lat', 'long'])
cleaned_data = data[['zipcode', 'price', 'sqft_living',
                     'condition', 'grade', 'floors',
                     'yr_built', 'yr_renovated']]
cleaned_data.head()

# %%
# Format categorical variables
cat_columns = ['condition', 'grade']
cleaned_data = hd.convert_categorical_variables(cleaned_data, cat_columns, True)
cleaned_data = hd.convert_categorical_variables(cleaned_data, ['floors'], False)
cleaned_data.head()

# %%
cleaned_data.describe()

# %%
# Scale Variables data
log_sqft_living = np.log(cleaned_data['sqft_living'])
sqft_living = cleaned_data['sqft_living']
# log_price = np.log(cleaned_data['price'])


# scaled_sqft_living = (log_sqft_living-np.mean(log_sqft_living))/np.sqrt(np.var(log_sqft_living))
# scaled_sqft_living = (log_sqft_living-min(log_sqft_living))/(max(log_sqft_living)-min(log_sqft_living))
scaled_sqft_living = (sqft_living-min(sqft_living))/(max(sqft_living)-min(sqft_living))
# scaled_price = (log_price-min(log_price))/(max(log_price)-min(log_price))
# Scale prices to K$
scaled_price = cleaned_data['price'].apply(lambda x: x/1000)
scaled_price = cleaned_data['price']

yr_renovated = cleaned_data['yr_renovated'].apply(lambda x: int(x.year))
log_yr_renovated = np.log(yr_renovated)
yr_built = cleaned_data['yr_built'].apply(lambda x: int(x.year))

scaled_yr_renovated = (log_yr_renovated-min(log_yr_renovated))/(max(log_yr_renovated)-min(log_yr_renovated))

data_fin = pd.DataFrame([])
# data_fin['sqft_living'] = scaled_sqft_living
data_fin['sqft_living'] = cleaned_data['sqft_living']
data_fin['price'] = scaled_price
data_fin['yr_renovated'] = scaled_yr_renovated
data_fin['yr_built'] = yr_built

scaled_data = cleaned_data.drop(['sqft_living', 'price', 'yr_renovated', 'yr_built'], axis=1)
# scaled_data = cleaned_data.drop('sqft_living', axis=1)
scaled_data = pd.concat([scaled_data, data_fin], axis=1)

scaled_data.head()


# %%
zipcode = 98072
data_per_zipcode = scaled_data.loc[scaled_data['zipcode'] == zipcode]
# print(data_per_zipcode)

cond_columns = hd.get_prefixed_column_names(cleaned_data, 'con_')
grade_columns = hd.get_prefixed_column_names(cleaned_data, 'gra_')
# floor_columns = hd.get_prefixed_column_names(cleaned_data, 'flo_')

predictors = data_per_zipcode.drop(['price', 'zipcode', 'floors', 'yr_built', 'yr_renovated'] + grade_columns + cond_columns, axis=1)
pred_sum = "+".join(predictors.columns)
formula = "price~" + pred_sum

# %%
model = ols(formula=formula, data=data_per_zipcode).fit()
model.summary()


# %%
result = hd.stepwise_selection(predictors, data_per_zipcode['price'], verbose=True)
print('resulting features:')
print(result)

# %%


# %%
zipcode = 98199
cat_variables = ['grade', 'condition']
cleaned_data = hd.convert_categorical_variables(data, cat_variables, True)

print(cleaned_data.head())
cond_columns = hd.get_prefixed_column_names(cleaned_data, 'con_')
grade_columns = hd.get_prefixed_column_names(cleaned_data, 'gra_')
print(cond_columns)

data_per_zipcode = cleaned_data.loc[cleaned_data['zipcode'] == zipcode]
# print(data_per_zipcode)
# predictors_columns = ['sqft_living', 'sqft_lot', 'sqft_living15'] + cat_variables
predictors_columns = ['sqft_living', 'sqft_lot', 'sqft_living15'] + grade_columns, cond_columns
predictors = data_per_zipcode[predictors_columns]

print(predictors)
pred_sum = "+".join(predictors.columns)
formula = "price~" + pred_sum

# %%
model = ols(formula=formula, data=data_per_zipcode).fit()
model.summary()

# %%
model.predict()
# result = stepwise_selection(predictors, data_fin["mpg"], verbose = True)
# print('resulting features:')
# print(result)

# %%
#from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

print(predictors.columns)
linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select=2)
selector = selector.fit(predictors, data_per_zipcode['price'])

# %%
selector.support_

# %%
selector.ranking_

# %%
estimators = selector.estimator_
print(estimators.coef_)
print(estimators.intercept_)
