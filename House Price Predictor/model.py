#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python
# coding: utf-8


# # ML Project

# In[335]:

# In[7]:


import pandas as pd


# In[336]:

# In[8]:


housing = pd.read_csv("data.csv")


# In[337]:

# In[9]:


housing.head()


# In[338]:

# In[10]:


housing.info()


# In[339]:

# In[11]:


housing['LSTAT'].value_counts()


# In[340]:

# In[12]:


housing.describe()


# In[341]:

# %matplotlib inline<br>
# import matplotlib.pyplot as plt<br>
# housing.hist(bins=50, figsize = (20, 15))

# # Train Test Splitting<br>
# 

# In[342]:

# In[13]:


import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[343]:

# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


# In[344]:

# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[345]:

# In[16]:


strat_train_set['CHAS'].value_counts()


# In[346]:

# In[17]:


strat_test_set['CHAS'].value_counts()


# In[347]:

# In[18]:


housing = strat_train_set.copy()


# # Looking for Correlation

# In[348]:

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[349]:

# In[20]:


corr_matrix.info()


# In[350]:

# In[21]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "ZN", "RM", "LSTAT"]
# scatter_matrix(housing[attributes], figsize=(12, 8))


# In[351]:

# In[22]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha = 0.8)


# Trying out combinations

# In[352]:

# housing["TAXRM"] = housing["TAX"]/housing["RM"]<br>
# corr_matrix = housing.corr()<br>
# corr_matrix['MEDV'].sort_values(ascending=False)

# In[353]:

# housing.plot(kind="scatter", x="TAXRM", y= "MEDV")

# In[354]:

# In[23]:


housing.info()


# In[355]:

# In[24]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# In[356]:

# In[25]:


housing.info()


# Missing Attributes

# To take care of missing attributes, you have three option:<br>
#     1. Get rid of the missing data points(If few data points are missing)<br>
#     2. Get rid of the whole attribute(When correlation is not matter)<br>
#     3. Set the value to some value (0, mean or median)

# In[357]:

# In[26]:


a= housing.dropna(subset="RM") #Option 1
a.shape
#Note that original data will be unchanged


# In[358]:

# In[27]:


housing.drop("RM", axis=1).shape #Option 2
#Note that original data will be unchanged


# In[359]:

# In[28]:


median = housing["RM"].median() #Option 3
housing["RM"].fillna(median)
#Note that original data will be unchanged


# In[360]:

# In[29]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
imputer.statistics_


# In[361]:

# In[30]:


X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()


# In[362]:

# In[31]:


housing_tr.info()


# Scikit-learn Design

# Primarily, three types of objects<br>
#  <br>
#   1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters.<br>
# <br>
#   2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.<br>
# <br>
#   3. Predictors - Linear Regression model is an example of predictors. fit() and predict() are two common funtions. It also gives score() function which will evaluate the predictions

# #Creating Pipelines

# Primarily, two types of feature scaling methods:<br>
# 1. Min-max scaling (Normalization):<br>
#     (value - min)/(max - min)<br>
#     Sklearn provides a class called MinMaxScaler for this<br>
# <br>
# 2. Standardization:<br>
#     (value - mean)/std<br>
#     sklearn provides a class called StandardScaler for this

# In[363]:

# In[32]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])


# In[364]:

# In[33]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[365]:

# In[34]:


housing_num_tr.shape


# Selecting a desired model

# In[366]:

# In[35]:


housing_labels.info


# In[367]:

# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# model = LinearRegression()<br>
# model = DecisionTreeRegressor()

# In[37]:


model = RandomForestRegressor()


# In[38]:


model.fit(housing_num_tr, housing_labels)


# In[368]:

# In[39]:


some_data = housing.iloc[:5]


# In[369]:

# In[40]:


some_labels = housing_labels.iloc[:5]


# In[370]:

# In[41]:


prepared_data = my_pipeline.transform(some_data)


# In[371]:

# In[42]:


model.predict(prepared_data)


# In[372]:

# In[43]:


list(some_labels)


# Evaluating the model

# In[373]:

# In[44]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[374]:

# In[45]:


rmse


# Using better evalution technique - Cross Validation

# In[375]:

# In[46]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)


# In[376]:

# In[47]:


rmse_scores


# In[377]:

# In[48]:


def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[378]:

# In[49]:


print_scores(rmse_scores)


# Saving The Model

# In[379]:

# In[50]:


from joblib import dump, load
dump(model, 'model.joblib')


# Testing the data on test data

# In[380]:

# In[51]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[381]:

# In[52]:


final_rmse

