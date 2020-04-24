import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd 
import matplotlib
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataIn = pd.read_csv ('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

# unitedStates = dataIn.query('Country_Region == "US"') #filter to just united states
unitedStates = dataIn.query('Province_State == "Washington"')
unitedStates = unitedStates[unitedStates.columns[10:]] #remove unnecessary columns
unitedStates = unitedStates.T # transpose the data set
unitedStates.columns = unitedStates.iloc[0] # uses labels as the column headers
unitedStates = unitedStates.iloc[1:, :] # remove the first row of column headers
# unitedStates.reset_index()
# unitedStates = unitedStates.reindex(unitedStates.index.drop('Combined_Key'))
# # unitedStates = unitedStates.loc[:, (unitedStates.sum() > 10000)] #filter to cities that add up to more than 10000
unitedStates['US Total'] = unitedStates.sum(axis=1) # add total column field
unitedStatesTotal = unitedStates.iloc[:, [-1]] # filter to just the total column
print(unitedStates.head())

modelUnitedStates = unitedStatesTotal
modelUnitedStates.index = range(0, len(modelUnitedStates))

x = modelUnitedStates.index
x = np.array(x).reshape(-1,1) # need to reshape the index data because it expects multiple columns to look at. not just one.
y = modelUnitedStates.iloc[:, 0]
# print(x)
# print(y)

### Linear Regression Example
# model = LinearRegression()
# model = model.fit(x, y)

# intercept = model.intercept_
# slope = model.coef_
# r_sq = model.score(x, y)
# print('intercept: ', intercept)
# print('slope: ', slope)
# print('r^2: ', r_sq)

### Polynomial Regression Example
# transformer = PolynomialFeatures(degree=2, include_bias=False) #transform the input array to contain the additional column(s) with the values of x^2 and eventually more.
# transformer.fit(x) # need to fit it before applying to the model
# x_transformed = transformer.transform(x) # modify the input

# # model = LinearRegression(fit_intercept=False).fit(x_transformed, y)
# model = LinearRegression().fit(x_transformed, y)

# intercept = model.intercept_
# slope = model.coef_
# r_sq = model.score(x_transformed, y)
# print('intercept: ', intercept)
# print('coefficient(s): ', slope)
# print('r^2: ', r_sq)

### stats oriented example
transformer = PolynomialFeatures(degree=2, include_bias=False) #transform the input array to contain the additional column(s) with the values of x^2 and eventually more.
transformer.fit(x) # need to fit it before applying to the model
x_transformed = transformer.transform(x) # modify the input
x_transformed = sm.add_constant(x_transformed)

model = sm.OLS(y, x_transformed)
results = model.fit()
predictions = results.predict()

# unitedStatesTotal.plot()
# predictions.plot()
# pyplot.show()

#set the interval lines for the model
prstd, iv_l, iv_u = wls_prediction_std(results)

fig, ax = pyplot.subplots(figsize=(8,6))

ax.plot(x, y, 'o', label="data")
# ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, results.fittedvalues, 'r', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best')
pyplot.show()

# print(results.summary())

# unitedStatesTotal.plot()
# pyplot.show()

# print(unitedStates.head())
# print(unitedStatesTotal.head())

# print(type(unitedStates))
# print('done')