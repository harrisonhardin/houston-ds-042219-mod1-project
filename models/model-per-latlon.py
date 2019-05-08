# %% [markdown]
# # Build model based on nearest properties

# %%
import housing_data as hd
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np

%load_ext autoreload
%autoreload 2

# %%
# Load dataset
data = hd.load_housing_data(with_cat_columns=False)

# %%
# Try to get Avg price per sqft base on nearest neighbors within radius (in km)
property_ds = data.iloc[0]  # Selected property
radius = 1  # 1km around selected property.

# Retrieve nearest properties
closet_properties_df = hd.get_closest_properties(data, property_ds, radius)

# Calculate Avg price per sqft and compare it against global Avg price/sqft
print('Average price/sqft_living : ', closet_properties_df['price'].mean()/closet_properties_df['sqft_living'].mean())
print('Global Average price/sqft_living : ', data['price'].mean()/data['sqft_living'].mean())

# %% [markdown]
# **Conclusion** : Now that we are able to calculate the Avg price around the
# property, we can generalise the calculus to the entire dataset.

# %%
# Run the prediction on a smaller dataset as the process take ages
# (~20mins depending on your machine).
enriched_data = data.copy()
sample_data = enriched_data[:1000]
sample_data['price_sqft'] = hd.get_price_per_sqft_living(sample_data)
sample_data.head()


# %% [markdown]
# # Building the model
# We are now going to try to run a simple Regression against our dataset

# %% [markdown]
# ## Convert categorical variables
# %%
# We are now going to try to run a simple Regression against our dataset
cat_variables = ['grade', 'condition']
cleaned_data = hd.convert_categorical_variables(sample_data, cat_variables, False)

# %% [markdown]
# ## Scale / Normalise variables

# %%
# Scale Variables data
log_sqft_living = np.log(cleaned_data['sqft_living'])
log_price_sqft = np.log(cleaned_data['price_sqft'])

# Scaling the variables
scaled_sqft_living = (log_sqft_living-np.mean(log_sqft_living))/np.sqrt(np.var(log_sqft_living))
scaled_price_sqft = (log_price_sqft-np.mean(log_price_sqft))/np.sqrt(np.var(log_price_sqft))

data_fin = pd.DataFrame([])
data_fin['sqft_living'] = scaled_sqft_living
data_fin['price_sqft'] = scaled_price_sqft

scaled_data = cleaned_data.drop(['sqft_living', 'price_sqft'], axis=1)
scaled_data = pd.concat([scaled_data, data_fin], axis=1)

scaled_data.head()


# %%
# Build formula
formula = 'price ~ sqft_living * price_sqft -1'

# %%
# Run simple prediction
model = ols(formula=formula, data=cleaned_data).fit()
model.summary()
