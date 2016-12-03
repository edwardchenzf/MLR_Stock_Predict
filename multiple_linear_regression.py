# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('zacks_custom_screen_2016-11-30.csv')
X = dataset.iloc[:, 2:10].values
y = dataset.iloc[:, 10].values

#define the number of rows as const from dataset
NUM_ROWS = 2718

'''# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap (by reducing one dummy variable, but library already take care of it. It is to emphasize the importance)
X = X[:, 1:]'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# y=b0+b1*x+....  add b0 in front
X = np.append(arr = np.ones((NUM_ROWS,1)).astype(int), values = X, axis =1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
# a simple ordinary least squares model
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          2.0717      0.440      4.712      0.000         1.210     2.934
x1             2.0717      0.440      4.712      0.000         1.210     2.934
x2         -3.874e-05   1.03e-05     -3.776      0.000     -5.89e-05 -1.86e-05
x3           4.75e-08   8.84e-08      0.538      0.591     -1.26e-07  2.21e-07
x4             0.1541      0.589      0.261      0.794        -1.001     1.309
x5             0.0727      0.019      3.844      0.000         0.036     0.110
x6            -0.5723      0.210     -2.724      0.006        -0.984    -0.160
x7            -0.0003      0.001     -0.235      0.814        -0.003     0.002
==============================================================================
"""
#remove x7
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          2.0644      0.438      4.708      0.000         1.205     2.924
x1             2.0644      0.438      4.708      0.000         1.205     2.924
x2         -3.877e-05   1.03e-05     -3.781      0.000     -5.89e-05 -1.87e-05
x3          4.755e-08   8.84e-08      0.538      0.591     -1.26e-07  2.21e-07
x4             0.1582      0.589      0.269      0.788        -0.996     1.313
x5             0.0728      0.019      3.850      0.000         0.036     0.110
x6            -0.5727      0.210     -2.727      0.006        -0.985    -0.161
==============================================================================

"""

#remove x4
X_opt = X[:, [0, 1, 2, 3, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          2.1440      0.323      6.636      0.000         1.510     2.778
x1             2.1440      0.323      6.636      0.000         1.510     2.778
x2         -3.888e-05   1.02e-05     -3.794      0.000      -5.9e-05 -1.88e-05
x3          4.937e-08   8.81e-08      0.560      0.575     -1.23e-07  2.22e-07
x4             0.0727      0.019      3.849      0.000         0.036     0.110
x5            -0.5688      0.209     -2.715      0.007        -0.980    -0.158
==============================================================================

"""
X_opt = X[:, [0, 1, 2, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""

==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          2.1688      0.320      6.776      0.000         1.541     2.796
x1             2.1688      0.320      6.776      0.000         1.541     2.796
x2         -3.629e-05   9.14e-06     -3.968      0.000     -5.42e-05 -1.84e-05
x3             0.0726      0.019      3.845      0.000         0.036     0.110
x4            -0.5722      0.209     -2.733      0.006        -0.983    -0.162
==============================================================================

"""