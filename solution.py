import os

import matplotlib.pyplot as plt
import numpy as np
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)


def plot_data(x: np.array, y: np.array, title: str):
    plt.scatter(x, y, c=y)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('rating')
    plt.ylabel('salary')
    plt.show()


def print_coef(coefs):
    for i in range(1, len(coefs)):
        coef = coefs[i]
        print(coef, end=", ")


def corr_heatmap(matrix, title, ticks):
    plt.imshow(matrix, cmap='coolwarm')
    # add a color bar
    plt.colorbar()
    plt.gcf().set_size_inches(7, 7)
    # add x-ticks, y-ticks and a label
    x_y_ticks = ticks
    plt.xticks(range(len(matrix.columns)), x_y_ticks, fontsize=12, rotation=90)
    plt.yticks(range(len(matrix.columns)), x_y_ticks, fontsize=12, rotation=90)
    plt.title(title, fontsize=16)

    labels = matrix.values
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            plt.text(i, j, '{:.3f}'.format(labels[j, i]), ha='center', va='center', color='black')

    plt.show()


def remove_variable(df, variables):
    for variable in variables:
        df.pop(variable)
    return df


def find_lowest_MAPE(subsets, X_train, X_test, y_train, y_test):
    all_mape_list_ = []
    min_mape = 100000
    subset_name = ''
    for subset in subsets:
        X_subset = X_train.drop(columns=subset)
        X_subset_test = X_test.drop(columns=subset)
        model = LinearRegression()
        model.fit(X_subset, y_train)
        predictions_test = model.predict(X_subset_test)
        curr_comb_mape = mape(y_test, predictions_test)

        if curr_comb_mape <= min_mape:
            min_mape = curr_comb_mape
            subset_name = subset

    return subset_name, min_mape


# read data
data = pd.read_csv('../Data/data.csv')
# Adding the intercept column to data
data['intercept'] = 1
# write your code here
y = data.salary
X = data[['intercept', 'rating']]
# plot_data(x=np.array(X['rating'].tolist()), y=np.array(y.tolist()), title='salary based on rating')


X = X ** 3
# Splitting the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

predictions_train = sklearn_model.predict(X_test)

MAPE = mape(y_true=y_test, y_pred=predictions_train)
# print(round(MAPE, 5))

# Based on the scattered plot of rating vs salary we notice the relationship between the two variables
# is different from linear and look like a polynomial function raising the rating variable by several
# degrees ( 2, 3 , 4 ) improved the score found with the MAPE


# We have used only one independant variable.
# Now we are going to use multiple varible in our model to make more accurate predictions

# In this bit of code I add the intercept column into the dataframe to then put it on the X dataframe
data = pd.read_csv('../Data/data.csv')
data['intercept'] = 1
intercept_col = data.pop('intercept')
y_col = 'salary'
y = data[y_col]
X = data[data.columns.drop(y_col)]
X.insert(0, 'intercept', intercept_col)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# make sure to print one of the sets to get a hand on the data we are working with
# print(X_train)

sklearn_model.fit(X_train, y_train)
preds = sklearn_model.predict(X_test)
MAPE_Test = mape(y_true=y_test, y_pred=preds)
# print(MAPE_Test)
coefs = sklearn_model.coef_.tolist()
# print_coef(coefs)

# A linear regression model with many variables might have some that are correlated
# Which can decrease the performance of the model, we must perform an import step, which is to check the model
# for multicollinearity and exclude the variables with a strong correlation

# Here is the correlation matrix
corr_data = data[['rating', 'draft_round', 'age', 'experience', 'bmi']]
ticks = ['rating', 'salary', 'draft_round', 'age', 'experience', 'bmi']
matrix_corr = corr_data.corr()
# corr_heatmap(matrix_corr, 'Salary based on variables', ['rating', 'dr_rd', 'age', 'exp', 'bmi'])
# from the heat map we can see that the variables with the highest correlation are (rating, age, experience)
# print(matrix_corr)

# First, try to remove each of the three variables
# Second, remove each possible pair of these three variables.
# The lowest MAPE is found by removing the experience and age variable
data = pd.read_csv('../Data/data.csv')
X, y = data.drop(columns=['salary']), data["salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
corr = X.corr()
corr_list = corr[corr != 1][corr > 0.2].dropna(how='all').index.to_list()  # ['rating', 'age', 'experience']
subsets = [['rating'], ['age'], ['experience'], ['rating', 'age'], ['rating', 'experience'], ['age', 'experience']]
subset, lowest_mape = find_lowest_MAPE(subsets, X_train, X_test, y_train, y_test)
# print(subset)

# We are going to fit the model with the correlated variables removed
X = X.drop(subset, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
sklearn_model.fit(X_train, y_train)
preds = sklearn_model.predict(X_test)
median_value = np.median(y_train)
preds[preds < 0] = 0
MAPE_median = mape(y_true=y_test, y_pred=preds)
print(round(MAPE_median, 5))
# Deal with negative values, we either will turn them into 0 or into the median value of y_train
# It turns out that transforming them into 0 give us a better MAPE score
