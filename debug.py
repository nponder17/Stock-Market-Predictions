import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
#import pandas_ta as ta

def main(): 
        fileName = 'stockMarket.csv'
        stockDataFrame = readFile(fileName)

        stockDataFrame = sortData(stockDataFrame)

        computedData = []

        for column in stockDataFrame.columns:
            if 'Price' in column:  #access _Price columns only
                prefix = column.replace('_Price', '')
                
                #Cleans up commas where needed
                computedData = []
                stockDataFrame = checkIfNeedToRemoveCommas(stockDataFrame, column)
                computedData = computeMovingAverage(stockDataFrame, column, prefix, computedData)
                computedData = getPreviousClose(stockDataFrame, column, prefix, computedData)
                computedData = computeStatistics(stockDataFrame,column,prefix,computedData)
        
                modelPredictions = getModelPredictions(stockDataFrame,column,computedData)



        computedDataFrame = pd.concat(computedData, axis = 1)
        stockDataFrame = pd.concat([stockDataFrame,computedDataFrame], axis = 1)
        #remove above comment to add computedData[] to stockDataFrame[]

        #print(stockDataFrame)
        #print(computedData)

        #print(modelPredictions)





def computeMovingAverage(stockDataFrame, column, prefix, computedData):
    # List of rolling window sizes for moving averages
    rollingWindows = [7, 14, 30, 200]

    #computedData = []
    for window in rollingWindows:
        rollingMean = (stockDataFrame[column].rolling(window, min_periods = 1).mean())
        rollingMean.name = (prefix +  " " + str(window) + " day average")
        computedData.append(rollingMean)

    return computedData

def checkIfNeedToRemoveCommas(dataframe, column):
    if dataframe[column].dtype == object:
        dataframe[column] = dataframe[column].str.replace(',','').astype(float)

    return dataframe

def sortData(stockDataFrame):
    #Converts string dates to datetime objects
    stockDataFrame['Date'] = pd.to_datetime(stockDataFrame['Date'],dayfirst=True)
        
    #Sorts dates in ascending order (helps with computing the moving average correctly)
    stockDataFrame = stockDataFrame.sort_values(by='Date', ascending=True)

    return stockDataFrame

#Function will read in file and return it as a pandas dataframe
def readFile(fileName):
    dataFrame = pd.read_csv(fileName)
    return dataFrame 

def getPreviousClose(stockDataFrame, column, prefix, computedData):
    #list of previous day count
    previousDay = [1,2,3]

    for day in previousDay:
        prevDay = stockDataFrame[column].shift(day)
        prevDay.name = (prefix + " " + str(day) + " days previous")

        computedData.append(prevDay)

    return computedData

def computeStatistics(stockDataFrame,column,prefix,computedData):

    dayChange = stockDataFrame[column].diff()
    print(type(computedData))
    DAYS_TO_AVERAGE_OVER = 14
    computedData['dayGain'] = (np.where(dayChange > 0, dayChange, 0))
    computedData['dayLoss'] = (np.where(dayChange < 0, dayChange, 0))
    computedData['avgGain'] = pd.Series(computedData['dayGain'].rolling(DAYS_TO_AVERAGE_OVER, min_periods = 1).mean())
    computedData['avgLoss'] = pd.Series(computedData['dayLoss'].rolling(DAYS_TO_AVERAGE_OVER, min_periods = 1).mean())

    #Calculate RS
    #RS = average gain/ average loss

    computedData['rs'] = computedData['avgGain']/computedData['avgLoss']
    #rs.name = (prefix+ ' RS')

    #Calculate RSI
    computedData['RSI'] = 100 - (100 / (1 + computedData['rs']))
    #RSI.name = (prefix + ' RSI')

    #moving average convergence divergence
    computedData['MACD'] = \
        stockDataFrame[column].ewm(span=12, adjust=False).mean() - \
        stockDataFrame[column].ewm(span=26, adjust=False).mean()
    
    computedData['signalLine'] = computedData['MACD'].ewm(span = 9, adjust = False).mean()

    #computedData.extend([dayGain, dayLoss, avgGain, avgLoss, rs, RSI, MACD, signalLine])

    return computedData

def getModelPredictions(stockDataFrame,column,computedData):

    X = computedData
    print(type(X))

    print(computedData['RSI'])
    exit()
    y = stockDataFrame[column]

    model = LinearRegression()
    o1 = model.fit(X,y)
    print(o1)

    o2 = modelPredictions = model.predict(X,y)
    print(o2)

    
    return modelPredictions

#from sklearn.linear_model import LinearRegression
#X = stockDataFrame[['RSI', 'SMA40', 'SMA100', 'Volume']]
#y = stockDataFrame['Close']

#model = LinearRegression()
#model.fit(X, y)
#predictions = model.predict(X)




if __name__ == "__main__":
    main()