def y_train_f(x):
    y_train = dataset.loc[x:x, "1960":"2015"].values
    y_train = np.squeeze(np.asarray(y_train))  # zmiana matrixa na array
    return y_train

def roznica_procentowa(x, y):
    return abs(((x*100)/y)-100)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("WorldPopulation.csv")

dataset.dropna( axis =1, inplace = True, thresh = 0.2)
# dataset.fillna(axis = 0, inplace = True, value = dataset.mean())

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(dataset.iloc[0:, 4:])
dataset.iloc[0:, 4:] = imputer.transform(dataset.iloc[0:, 4:])

X_train = pd.DataFrame(np.arange(1960, 2016)).values
X_test = pd.DataFrame()
y_test = dataset.loc[:, "2016"]

from sklearn.linear_model import LinearRegression

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X_train)


# podglądwyników
"""
# X_grid = np.arange(min(X_train), max(X_train), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_train, y_train_f(x), color="red")
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='green')
# plt.title("polynomial regression wyniki populacji")
# plt.xlabel("lata")
# plt.ylabel("populacja")
# plt.show()
"""

# sprawdzenie roku 2016 dla każdego z państw

for x in range(0, 217):
    y_train_f(x)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_train_f(x))
    temp = lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
    # print(f"{dataset.loc[x, 'Country']} {temp}")
    X_test[x] = temp


print(roznica_procentowa(X_test, y_test))



# ...:
"""
jaki stopień wielomianu daje najdokładniejsze wyniki
od któego roku otrzymuje się najdokładniejsze wyniki
"""