# %% [markdown]
# # Build model based on nearest properties that predict your property value
#
# It might be usefull to know your neighbourhood avg price in order to predic
# the value of the existing property. So the question is How does that work ?.
#
# Table of Contents
# 1. [Getting our nearest properties](#Getting-our-nearest-properties)
# 2. [Running nearest properties avg price/sqft against dataset](#Running-nearest-properties-avg-price-per-sqft-against-dataset)
# 3. [Convert categorical variables](#Convert-categorical-variables)
# 4. [Scale and normalise variables](#Scale-and-normalise-variables)
# 5. [Building the model](#Building-the-model)
# 6. [Conclusion](#Conclusion)

# %%
import housing_data as hd
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## Getting our nearest properties
#
# It might be usefull to know your neighbourhood avg price in order to predic
# the value of the existing property. For that we need to come up with an
# algorithm that retrieves those nearest properties.
# see housing_data module for more details.


# %%
# Load dataset
data = hd.load_housing_data(with_cat_columns=False)

# %%
# Try to get Avg price per sqft base on nearest neighbors within radius (in km)
property_ds = data.iloc[0]  # Selected property
radius = 1  # 1 mile around selected property.

# Retrieve nearest properties
closet_properties_df = hd.get_closest_properties(data, property_ds, radius)

# Calculate Avg price per sqft and compare it against global Avg price/sqft
print('Average price/sqft_living : ', closet_properties_df['price'].mean()/closet_properties_df['sqft_living'].mean())
print('Global Average price/sqft_living : ', data['price'].mean()/data['sqft_living'].mean())

# %% [markdown]
# **Conclusion** : Now that we are able to calculate the Avg price around the
# property, we can generalise the calculus to the entire dataset.

# %%
# ## Running nearest properties avg price per sqft against dataset
# Run the prediction on a smaller dataset as the process take ages
# (~20mins depending on your machine).
enriched_data = data.copy()
sample_data = enriched_data[:1000]

# %%
# Enriched the data by adding the avg sqft price of neighbours
sample_data['price_sqft'] = hd.get_price_per_sqft_living(sample_data)
sample_data.head()


# %% [markdown]
# ## Convert categorical variables
# %%
# We are now going to try to run a simple Regression against our dataset
cat_variables = ['grade', 'condition']
cleaned_data = hd.convert_categorical_variables(sample_data, cat_variables, False)

# %% [markdown]
# ## Scale and normalise variables

# %%
# Plotting variables to see distribution and skewness
x_cols = ['price', 'sqft_living', 'price_sqft']
pd.plotting.scatter_matrix(cleaned_data[x_cols], figsize=(10,12));

# %% [markdown]
# **Conclusion** : We can see that these variables are not normally distriburted
# Some log normalisation is needed in order to remove skewness.


# %%
# Scale Variables data
log_sqft_living = np.log(cleaned_data['sqft_living'])
log_price_sqft = np.log(cleaned_data['price_sqft'])

# Scaling the variables
scaled_sqft_living = (log_sqft_living-min(log_sqft_living))/(max(log_sqft_living)-min(log_sqft_living))
scaled_price_sqft = (log_price_sqft-min(log_price_sqft))/(max(log_price_sqft)-min(log_price_sqft))

data_fin = pd.DataFrame([])
data_fin['sqft_living'] = scaled_sqft_living
data_fin['price_sqft'] = scaled_price_sqft

scaled_data = cleaned_data.drop(['sqft_living', 'price_sqft'], axis=1)
scaled_data = pd.concat([scaled_data, data_fin], axis=1)


# %% [markdown]
# # Building the model
# We are now going to try to run a simple Regression against our dataset

# %%
# Build formula
# Notes that we are especting a corrolation between sqft_living and price_sqft
formula = 'price ~ sqft_living * price_sqft -1'

# %%
# Run simple prediction
model = ols(formula=formula, data=scaled_data).fit()
model.summary()

# %% [markdown]
# **Observations** : The Adj. R-squared is pretty high and our variables coef
# p-values are low. This looks much better than the zipcode model.

# %% [markdown]
# # Conclusion
# We saw that the model built based on lat/lon proximity is more accurate
# than the one based on zipcode. This might be true because of price variation
# within a specific zipcode.
# We are now going to try to run a simple Regression against our dataset
