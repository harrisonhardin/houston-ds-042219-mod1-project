# %% markdown
# # Multicollinearity of Features
#
# The objective here is to identify multicollinearity between our features.
#
# # Table of Contents - TBD
# 1. [Step 1: Dealing with NaN values](#Step-1:-Dealing-with-NaN-values)
# 2. [Step 2: Converting data types](#Step-2:-Converting-data-types)
# 3. [Step 3: Categorical variables](#Step-3:-Categorical_variables)
# 3. [Step 4: Save our cleaned dataset for reusability](#Step-4:-Save-our-cleaned-dataset-for-reusability)
# 4. [Conclusion](#Conclusion)
# %% markdown
# ## Step 1: Identifying multicollinearity
# %%
# Set auto reload
%load_ext autoreload
%autoreload 2

# %%
# Reload our cleaned dataset and perform date typing
import pandas as pd
import housing_data as hd

df = pd.read_csv('cleaned_kc_house_data.csv')

# We can see that the datetime format is not kept. Let's fix it
date_columns = ['date', 'yr_built', 'yr_renovated']

df = hd.format_date_columns(df, date_columns)

df.head()
# %%
data_pred= df.iloc[:,3:]
data_pred.head()
# %%
abs(data_pred.corr()) > 0.75
# %%
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

fig = plt.figure(figsize=(40,40))
sns.heatmap(data_pred.corr(), center=0);

# %% markdown
# **Conclusion** We can see some collinearity between sqft_living and sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15 (At the top left corner - orange area)
# %%
from statsmodels.formula.api import ols
# %%
grade_columns = hd.get_prefixed_column_names(df, 'grade_')

print(grade_columns)

outcome = 'price'
x_cols = ['sqft_living', 'lat', 'long']

x_cols = ['sqft_living'] + grade_columns

predictors = '+'.join(x_cols)
formula = outcome + "~" + predictors
model = ols(formula=formula, data=df).fit()
model.summary()
# %%
data_per_zipcode = df[df['zip_98146'] == 1]

outcome = 'price'
x_cols = ['sqft_living', 'water_True']
predictors = '+'.join(x_cols)
formula = outcome + "~" + predictors
model = ols(formula=formula, data=data_per_zipcode).fit()
model.summary()
# %%
pd.plotting.scatter_matrix(data_per_zipcode[x_cols], figsize=(10,12));
# %%
import numpy as np

non_normal = ['sqft_living']

log_sqft_living = np.log(data_per_zipcode['sqft_living'])

scaled_sqft_living = (log_sqft_living-np.mean(log_sqft_living))/np.sqrt(np.var(log_sqft_living))

data_fin = pd.DataFrame([])
data_fin['sqft_living']= scaled_sqft_living

price = data_per_zipcode['price']

data_fin = pd.concat([price, data_fin])

data_ols = pd.concat([price, scaled_sqft_living], axis=1)

data_ols.head()
# for feat in non_normal:
#     data_per_zipcode[feat] = data_per_zipcode[feat].apply(lambda x: np.log(x))
#     data_per_zipcode_98146[feat] = data_per_zipcode_98146[feat].map(lambda x: np.log(x))
# pd.plotting.scatter_matrix(data_per_zipcode_98146[x_cols], figsize=(10,12));
# %%
outcome = 'price'
predictors = data_ols.drop('price', axis=1)
#predictors = predictors.drop("orig_3",axis=1)
pred_sum = "+".join(predictors.columns)
formula = outcome + "~" + pred_sum
# %%
model = ols(formula= formula, data=data_ols).fit()
model.summary()
# %%
pd.plotting.scatter_matrix(data_per_zipcode_98146[x_cols], figsize=(10,12));
# %%
outcome = 'price'
x_cols = ['sqft_living', 'lat', 'long', 'water_True']
predictors = '+'.join(x_cols)
formula = outcome + "~" + predictors
model = ols(formula=formula, data=data_per_zipcode_98146).fit()
model.summary()
# %%

# %%
selected_predictors = []
# %%
pd.plotting.scatter_matrix(data[x_cols], figsize=(10,12));
# %%
