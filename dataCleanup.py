import statsmodels as ss
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd 
import matplotlib
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataIn = pd.read_csv ('csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
# print(dataIn[0:5])

# unitedStates = dataIn[dataIn.Country_Region == "US"]
unitedStates = dataIn.query('Country_Region == "US"') #filter to just united states
unitedStates = unitedStates[unitedStates.columns[10:]] #remove unnecessary columns
unitedStates = unitedStates.T 
unitedStates.columns = unitedStates.iloc[0] # uses labels as the column headers
unitedStates = unitedStates.iloc[1:, :] # remove the first row of column headers
unitedStates.reset_index()
# unitedStates = unitedStates.reindex(unitedStates.index.drop('Combined_Key'))
# # unitedStates = unitedStates.loc[:, (unitedStates.sum() > 10000)] #filter to cities that add up to more than 10000
unitedStates['Sum Cities'] = unitedStates.sum(axis=1) # add total column field
unitedStatesTotal = unitedStates.iloc[:, [-1]] # filter to just the total column 

# x = unitedStatesTotal.index
# y = unitedStates.iloc[:, 1]

# model = LinearRegression()
# model = model.fit(x, y)

# r_sq = model.score(x, y)
# print(r_sq)

# print(unitedStates)
# print(list(unitedStates.columns[:10]))
# print(unitedStates.index[:3])

# unitedStates.plot()
# pyplot.show()

print(unitedStates.head())
# print(type(unitedStates))
# print('done')