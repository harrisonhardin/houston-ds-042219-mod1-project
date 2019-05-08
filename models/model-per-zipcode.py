# %% [markdown]
# # Build Regression Model based on data_per_zipcode
#
# As seen during our collinearity analysis it seems that zipcode might affect
# the property price.

# %% [markdown]
# # Build Regression Model based on data_per_zipcode to predict property price
#
# As seen during our collinearity analysis it seems that zipcode might affect
# the property price.
#
# So the question is How does that work ?.
#
# Table of Contents
# 1. [EDA dataset per zipcode](#EDA-dataset-per-zipcode)
# 4. [Scale and normalise variables](#Scale-and-normalise-variables)
# 5. [Building the model](#Building-the-model)
# 6. [Conclusion](#Conclusion)

# %% [markdown]
# # Prep-work

# %%
# Import useful librairies and set auto reload
import pandas as pd
import numpy as np
import housing_data as hd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from statsmodels.formula.api import ols

%load_ext autoreload
%autoreload 2


# %% [markdown]
# ##EDA dataset per zipcode

# %%
# Load dataset
data = hd.load_housing_data(with_cat_columns=False)
data.head()

# %%
# Plot property count per zipcode
g = sns.factorplot("zipcode", data=data, aspect=1.5, kind="count", color="b")
g.set_xticklabels(rotation=30)

# %% [markdown]
# It looks like we got enough data per zipcode.

# %% [markdown]
# **Display price against sqft_living**

# %%
# Split dataset in two group of zipcode for display purpose.
zipcodes = data['zipcode'].unique()
zipcodes.sort()
zipcodes_median = zipcodes[(len(zipcodes) // 2): (len(zipcodes) // 2) + 1][0]

data_set1 = data.loc[data['zipcode'] <= zipcodes_median]
data_set2 = data.loc[data['zipcode'] > zipcodes_median]

# %%
# Display price vs sqft_living on the first subset
plt.figure(figsize=(10,10))
g = sns.FacetGrid(data_set2, col="zipcode", col_wrap=3)
g.map(plt.scatter, "sqft_living", "price")
g.set(xlim=(0, 6000), ylim=(0, 3000000))
g.set_xlabels('')
g.set_ylabels('Property price')

# %% [markdown]
# **Conclusion** : The plots seem to be linear. Let's validate this hypothesis
# by building a model on a specific zipcode.

# %% [markdown]
# ## Scale and normalise variables

# %%
# Scale Variables data
log_sqft_living = np.log(data['sqft_living'])
scaled_sqft_living = (log_sqft_living-min(log_sqft_living))/(max(log_sqft_living)-min(log_sqft_living))

data_fin = pd.DataFrame([])
data_fin['sqft_living'] = scaled_sqft_living

scaled_data = data.drop(['sqft_living'], axis=1)
scaled_data = pd.concat([scaled_data, data_fin], axis=1)

# %% [markdown]
# # Building the model
# We are now going to try to run a simple Regression against our dataset

# %%
# Get sample data from specific zipcode
zipcode = 98072
data_per_zipcode = scaled_data.loc[scaled_data['zipcode'] == zipcode]

# %%
# Build formula
# Notes that we are especting a corrolation between sqft_living and price_sqft
formula = 'price ~ sqft_living'

# %%
# Run simple prediction
model = ols(formula=formula, data=data_per_zipcode).fit()
model.summary()


# %% [markdown]
# **Observations** : The Adj. R-squared is pretty low and our variables coef
# p-values are low. This doesn't look good.

# %% [markdown]
# # Conclusion
# We saw that the model built based on zipcode proximity is not really accurate
# We need to come up with a more precise model.
