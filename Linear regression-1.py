#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[55]:


df=pd.read_csv("mpg_raw.csv")


# In[56]:


df.head()


# # Data discription:
# 
# 1.mpg -It provides details about the mileage per gallon performances of various cars.(Our Target variable)
# 
# 2.cylinders -The number of cylinders in an engine directly influences its power output and fuel consumption
# *More Cylinders: Engines with more cylinders (e.g., V6, V8) tend to produce higher power and torque. However, they also consume more fuel.
# *Fewer Cylinders: Engines with fewer cylinders (e.g., inline-4, inline-3) are generally more fuel-efficient but may have lower power output.
# 
# 3.displacement - Displacement refers to the total volume swept by all the pistons inside an engine’s cylinders during one complete cycle (intake, compression, power, and exhaust).
# 
# 4.horsepower - It contributes to a car’s performance, it also affects fuel efficiency.
# 
# 5.weight - Weight of the car.
# 
# 6.acceleration- Accelerating too quickly without considering road conditions may lead to unnecessary braking and additional acceleration, affecting overall efficiency3.
# 
# 7.model_year - YEAR of yhe model.
# 
# 8.origin - Manufacturing place of the car.
# 
# 9.name - Name of the car

# In[57]:


df.shape


# In[58]:


df.dtypes


# In[59]:


df.info()


# In[60]:


df.describe()


# ----------------------------------------

# #### Value counts for each column

# In[61]:


df['mpg'].unique()


# In[62]:


df['cylinders'].unique()


# In[63]:


df['displacement'].unique()


# In[64]:


df['horsepower'].unique()


# In[65]:


df['weight'].unique()


# In[66]:


df['acceleration'].unique()


# In[67]:


df['model_year'].unique()


# In[68]:


df['origin'].unique()


# In[69]:


df['name'].unique()


# ---------------------------------------------

# #### handling null values

# In[70]:


df.isna().sum()


# In[71]:


df[df['horsepower'].isna()]


# #### replacing horsepower null values to mode of horsepower accordingly

# In[72]:


value_for_greater100 = df[(df['displacement']>100) & (df['displacement']<=200) & (df['weight']>2500)]['horsepower'].mode()[0]
value_for_greater100


# In[73]:


df.loc[(df['displacement'] > 100) & (df['horsepower'].isna()), 'horsepower'] = value_for_greater100


# In[74]:


value_for_lesser100 =df[(df['displacement']<=100) & (df['displacement']>84) & (df['weight']<2300) & (df['weight']>1800)]['horsepower'].mode()[0]
value_for_lesser100


# In[75]:


df.loc[(df['displacement'] <= 100) & (df['horsepower'].isna()), 'horsepower'] = value_for_lesser100


# ----------------------------------

# #### correlation visualization for each column with target attribute

# In[76]:


sns.scatterplot(x=df['mpg'],y=df['displacement'])
print('\033[1mMPG and Displacement are Negatively correlated\033[0m')


# In[77]:


sns.scatterplot(x=df['mpg'],y=df['horsepower'])
print('\033[1mMPG and horsepower are Negatively correlated\033[0m')


# In[78]:


sns.scatterplot(x=df['mpg'],y=df['weight'])
print('\033[1mMPG and weight are Negatively correlated\033[0m')


# In[79]:


sns.scatterplot(x=df['mpg'],y=df['acceleration'])
print('\033[1mAcceleration not too correlated with MPG\033[0m')


# In[80]:


sns.scatterplot(x=df['mpg'],y=df['weight'],hue=df['origin'])
print('\033[1mUSA cars are weigh more than Japan and Europe\033[0m')


# ####  Dropping name column since there is no relation with MPG column

# In[81]:


df.drop('name',axis=1,inplace=True)


# ### OUTLIERS

# 
# ----------------------------
# 
# #### detecting outliers of displacement

# In[82]:


# trimming the outliers of displacement
upperlimit_displacement = df['displacement'].quantile(0.99)
lowerlimit_displacement = df['displacement'].quantile(0.01)
print('upperlimit_displacement:', upperlimit_displacement)
print('lowerlimit_displacement:', lowerlimit_displacement)
# trimming the outliers of displacement
df_displacement = df.loc[(df['displacement']<= upperlimit_displacement) & (df['displacement']>=lowerlimit_displacement)]
print('before removing outliers:', len(df))
print('after removing outliers:', len(df_displacement))
print('outlier data:',len(df)-len(df_displacement))
df_displacement


# #### Replacing outliers of displacement with upperlimit and lowerlimit values,
# the values which are in outliers portion are very close to upperlimit and lowerlimit

# In[83]:


df.loc[(df['displacement']>upperlimit_displacement),'displacement']= round(upperlimit_displacement)


# In[84]:


df.loc[(df['displacement']<lowerlimit_displacement),'displacement']= round(lowerlimit_displacement)


# #### detecting outliers of horsepower

# In[85]:


# trimming the outliers of horsepower
upperlimit_horsepower = df['horsepower'].quantile(0.99)
lowerlimit_horsepower = df['horsepower'].quantile(0.01)
print('upperlimit_horsepower:', upperlimit_horsepower)
print('lowerlimit_horsepower:', lowerlimit_horsepower)
# trimming the outliers of horsepower
df_horsepower = df.loc[(df['horsepower']<= upperlimit_horsepower) & (df['horsepower']>=lowerlimit_horsepower)]
print('before removing outliers:', len(df))
print('after removing outliers:', len(df_horsepower))
print('outlier data:',len(df)-len(df_horsepower))
df_horsepower


# #### Replacing outliers of displacement with upperlimit and lowerlimit values,
# the values which are in outliers portion are very close to upperlimit and lowerlimit also traget value is moeover same

# In[86]:


df.loc[(df['horsepower']>upperlimit_horsepower),'horsepower']= round(upperlimit_horsepower)


# In[87]:


df.loc[(df['horsepower']<lowerlimit_horsepower),'horsepower']= round(lowerlimit_horsepower)


# 
# #### Detecting outliers of weight using boxplot visualization

# In[88]:


sns.boxplot(df['weight'])
print('\033[1mSeems no outliers detected in weight column\033[0m')


# In[89]:


# trimming the outliers of weight
upperlimit_weight = df['weight'].quantile(0.99)
lowerlimit_weight = df['weight'].quantile(0.01)
print('upperlimit_weight:', upperlimit_weight)
print('lowerlimit_weight:', lowerlimit_weight)
# trimming the outliers of weight
df_weight = df.loc[(df['weight']<= upperlimit_weight) & (df['weight']>=lowerlimit_weight)]
print('before removing outliers:', len(df))
print('after removing outliers:', len(df_weight))
print('outlier data:',len(df)-len(df_weight))
df_weight


# #### Replacing outliers of weight with upperlimit and lowerlimit values,
# the values which are in outliers portion are very close to upperlimit and lowerlimit.

# In[90]:


df.loc[(df['weight']>upperlimit_weight),'weight']= round(upperlimit_weight)


# In[91]:


df.loc[(df['weight']<lowerlimit_weight),'weight']= round(lowerlimit_weight)


# 
# ------------------------------------------
# 
# 
# ### Feature engineering
# 
# Categorical Encoding:
# Convert categorical variables into numerical representations.

# In[44]:


from sklearn.preprocessing import OneHotEncoder


# In[93]:


encoder=OneHotEncoder()
encoded_features = encoder.fit_transform(df[['origin']]).toarray()
df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['origin']))
df_encoded


# In[95]:


result = pd.concat([df, df_encoded], axis=1, join='inner')
result.head()


# #### Dropping origin column since there is no relation with MPG column

# In[99]:


result.drop('origin',axis=1,inplace=True)


# In[100]:


result.corr()


# In[101]:


df=result.copy()
df.head()


# ----------------------------------------------------
# 
# 
# 
# ### Splitting train and test data

# In[107]:


from sklearn.model_selection import train_test_split
Y=df['mpg']
X=df.drop(columns=['mpg'])
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=42) 
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))


# --------------------------------------------------
# 
# 
# 
# ### checking the train and test data's are coming from same population by two sample t test

# In[112]:


import scipy.stats as stats

for i,j in zip (X_train.columns,X_test.columns):
    x1=X_train[i].values
    x2=X_test[j].values
    t_statistic, p_value = stats.ttest_ind(x1, x2)
    print(f"Two-sample t-test: t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")


# In[113]:


y1=Y_train.values
y2=Y_test.values
t_statistic, p_value = stats.ttest_ind(y1, y2)
print(f"Two-sample t-test: t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")


# #### Since the p-values for most pairs of samples are greater than 0.05 (common significance level), we fail to reject the null hypothesis.
# Therefore, there is insufficient evidence to conclude that these samples come from different populations.
# 
# 
# 
# 
# 
# ----------------------------------

# ### gradient_descent -->It aims to reach the point of minimum error (cost) by adjusting the parameters step by step.

# In[114]:


def gradient_descent(x,y):
    slope=intercept=0
    iterations=3000
    learningrate=0.02
    n=len(x)
    for i in range(iterations):
        y_predicted = slope*x + intercept
        error=(1/n)*sum([val**2 for val in (y-y_predicted)])
        slope_dev=-(2/n)*sum(x*(y-y_predicted))
        intercept_dev=-(2/n)*sum(y-y_predicted)
        slope=slope-learningrate*slope_dev
        intercept=intercept-learningrate*intercept_dev
        print(f"slope{slope},intercept{intercept},error{error},iterations{i}")
   

x=df['cylinders']
y=df['mpg']
gradient_descent(x,y)


# ### OLS Ordinary least square method
# 
# Statistical method to find the unknown parameters(Slope,intercept, error) of linear regression model,by fitting regression best fit line and optimizing the slope and intercept where the error is sum of square of difference between y actual value and y-predicted value is  as small as possible.
# 

# In[115]:


import statsmodels.api as sm
constant = sm.add_constant(X_train)
result = sm.OLS(Y_train,constant).fit()
print(result.summary())


# ### Linear Regression:
# Linear regression models the relationship between a dependent variable (response) and one or more independent variables (predictors) using a linear equation.
# It assumes a linear relationship between the variables.

# #### Model fitting(multiple linear regresstion)

# In[116]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
lmodel=model.fit(X_train,Y_train)
y_predicted=lmodel.predict(X_test)
print(f"slope:{lmodel.coef_},intercept:{lmodel.intercept_}")
y_predicted


# In[117]:


from sklearn import metrics 
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(Y_test,y_predicted))  
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test, y_predicted))  
print('Root Mean Squared Error :', np.sqrt(metrics.mean_squared_error(Y_test, y_predicted)))


# In[118]:


from sklearn.metrics import r2_score
print('r2_score:', r2_score(Y_test,y_predicted)) 


# ### Model MSE:
# 
# #### Definition: 
# The model MSE quantifies the average squared difference between the predicted values and the actual target values (ground truth) for a regression model.
# #### Purpose:
# It helps assess how well the model fits the data. Lower MSE values indicate better model performance, as they indicate smaller prediction errors.
# #### Formula:
# MSE=n1​i=1∑n​(yi​−y^​i​)2
# 
# 
# ### Baseline MSE (Null Model MSE):
# 
# #### Definition:
# The baseline MSE represents the error if we use a simple baseline model (e.g., mean or median) that ignores any features or predictors.
# #### Purpose:
# It provides a reference point for comparison. If the model’s MSE is significantly lower than the baseline MSE, it indicates that the model is adding value beyond what the baseline provides.
# #### Formula (for a mean baseline):
# Baseline MSE=n1​i=1∑n​(yi​−yˉ​)2

# In[167]:


from sklearn.metrics import mean_squared_error
# Create a baseline (simple mean prediction)
mean_mpg = Y_train.mean()
y_baseline = [mean_mpg] * len(Y_test)
mse_baseline = mean_squared_error(Y_test, y_baseline)
# Calculate MSE for the model
mse_model = mean_squared_error(Y_test, y_predicted)
# Compare metrics
print(f"Model MSE: {mse_model:.2f}")
print(f"Baseline MSE: {mse_baseline:.2f}")
print("MSE is lower in our case, So the model fitting is good.")


# ### Assumptions of linear regression

# #### 1.Checking the residuals are normally distributed
# 
# Most of the residuals closely follow the reference line, it suggests that the residuals are normally distributed.

# In[158]:


residuals=result.resid
sm.qqplot(residuals,line='s')
plt.show()


# #### 2.Checking the residuals are independent & have homoskedasticity

# In[165]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
# Create a figure for residual plots
for i in ('cylinders','displacement','horsepower','weight','acceleration','model_year','origin_europe','origin_japan','origin_usa'):
    fig = plt.figure(figsize=(14, 8))
    fig = sm.graphics.plot_regress_exog(result,i, fig=fig)
    plt.show() 


# we can see that the points are plotted randomly spread or scattered. points or residuals are scattered around the ‘0’ line, there is no pattern and points are not based on one side so there’s no problem of heteroscedasticity.with the predictor variable ‘horsepower’ there’s no heteroscedasticity. 

# In[ ]:




