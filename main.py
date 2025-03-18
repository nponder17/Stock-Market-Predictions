import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils import plot_model


def main(): 
        fileName = 'stockMarket.csv'
        stockDataFrame = readFile(fileName)

        stockDataFrame = sortData(stockDataFrame)


        for column in stockDataFrame.columns:
            if 'Price' in column:  #access _Price columns only
                prefix = column.replace('_Price', '')
                
                #Cleans up commas where needed
                computedData = []
                stockDataFrame = checkIfNeedToRemoveCommas(stockDataFrame, column)
                computedData = computeMovingAverage(stockDataFrame, column, prefix, computedData)
                computedData = getPreviousClose(stockDataFrame, column, prefix, computedData)
                computedData = computeStatistics(stockDataFrame,column,prefix,computedData)

                #access volume column
                volumeColumns = processVolumeColumn(stockDataFrame, prefix)


                #exception for S&P where there is no volume column, if no volume column skip to the next iteration
                if volumeColumns is None:
                    continue

                #use correlation map to check feature correlation
                allFeatures = checkFeatureCorrelation(stockDataFrame[column], volumeColumns, computedData)
            

                #TensorFlow Predictions
                tensorFlowPredictions = getTensorFlowPredictions(allFeatures, prefix)

                

        #print(stockDataFrame)
        #print(computedData)
                


def processVolumeColumn(stockDataFrame, prefix):

    try:
        volumeColumn = stockDataFrame[prefix + '_Vol.']
        return volumeColumn
    
    except KeyError:
        print("Warning: Volume column does not exist for " , prefix)
        return None


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

def computeStatistics(stockDataFrame, column, prefix, computedData):
    dayChange = stockDataFrame[column].diff()
    
    DAYS_TO_AVERAGE_OVER = 14
    dayGain = np.where(dayChange > 0, dayChange, 0)
    dayLoss = np.where(dayChange < 0, dayChange, 1e-7)
    avgGain = pd.Series(dayGain).rolling(DAYS_TO_AVERAGE_OVER, min_periods=1).mean()
    avgLoss = pd.Series(dayLoss).rolling(DAYS_TO_AVERAGE_OVER, min_periods=1).mean()

    # Calculate RS & RSI
    rs = avgGain / avgLoss
    RSI = 100 - (100 / (1 + rs))

    # Compute MACD and Signal Line
    MACD = stockDataFrame[column].ewm(span=12, adjust=False).mean() - stockDataFrame[column].ewm(span=26, adjust=False).mean()
    signalLine = MACD.ewm(span=9, adjust=False).mean()

    computedData = pd.DataFrame({
        prefix + '_day_gain': dayGain,
        prefix + '_day_loss': dayLoss,
        prefix + '_avg_gain': avgGain,
        prefix + '_avg_loss': avgLoss,
        prefix + '_RS': rs,
        prefix + '_RSI': RSI,
        prefix + '_MACD': MACD,
        prefix + '_signal_line': signalLine
    })

    return computedData

def checkFeatureCorrelation(priceColumn, volumeColumn, computedData):

     nextDayPrice = priceColumn.shift(-1)

    #ensure that the vectors price and volume allign in shape with computed data matrix
     potentialFeatures = pd.concat([priceColumn, volumeColumn, computedData], axis=1).fillna(method='ffill').fillna(method='bfill')
    # Fill missing values before assigning to DataFrame
     nextDayPrice = nextDayPrice.fillna(method='ffill').fillna(method='bfill')

    # Now assign to DataFrame
     potentialFeatures['Next Day Price'] = nextDayPrice

     #print(potentialFeatures.shape)
     


     correlationMatrix = potentialFeatures.corr()

     #print(correlationMatrix)

     return potentialFeatures


def getTensorFlowPredictions(allFeatures, prefix):

    #create features for the model
    #since we will want to try to predict the next days price
    #we will drop the next day price and not include it in the features

    features = allFeatures.drop(columns = ['Next Day Price']).values

    priceChange = allFeatures["Next Day Price"].values - allFeatures[prefix + '_Price'].values

    targetGainLoss = (priceChange > 0).astype(int)
    targetPrice = allFeatures["Next Day Price"].values

    targets = np.column_stack((targetGainLoss, targetPrice))

    #check the shape of features and target
    #features should be 1243 x 10 (numerical)
    #target should be 1243 x 1 (binary)
    #print(features.shape)
    #print(target.shape)

    print("targetGainLoss shape:", targetGainLoss.shape)
    print("targetPrice shape:", targetPrice.shape)


    #convert features and target into tensor flow tensors

    featureTensor = tf.convert_to_tensor(features, dtype = tf.float32)
    targetTensor = tf.convert_to_tensor(targets, dtype = tf.float32)

    #check shape for debugging

    #print(featureTensor.shape)
    #print(targetTensor.shape)

    # Split data
    trainingSize = int(0.8 * len(featureTensor))
    featureTrain, featureTest = featureTensor[:trainingSize], featureTensor[trainingSize:]
    targetTrain, targetTest = targetTensor[:trainingSize], targetTensor[trainingSize:]


    # Normalize the feature data
    scaler = StandardScaler()
    featureTrain = scaler.fit_transform(featureTrain)  # Fit on train, transform train
    featureTest = scaler.transform(featureTest)  # Only transform test

    numSamplesTrain = featureTrain.shape[0]
    numSamplesTest = featureTest.shape[0]
    numFeatures = featureTrain.shape[1]
    timeSteps = 10

    # Make sure the number of samples is divisible by timeSteps
    numSamplesTrain = numSamplesTrain - (numSamplesTrain % timeSteps)
    numSamplesTest = numSamplesTest - (numSamplesTest % timeSteps)

    # Trim the feature and target arrays accordingly
    featureTrain = featureTrain[:numSamplesTrain]
    featureTest = featureTest[:numSamplesTest]
    targetTrain = targetTrain[:numSamplesTrain]
    targetTest = targetTest[:numSamplesTest]

    # Now reshape the features and targets
    featureTrain = tf.reshape(featureTrain, (numSamplesTrain // timeSteps, timeSteps, featureTrain.shape[1]))
    featureTest = tf.reshape(featureTest, (numSamplesTest // timeSteps, timeSteps, featureTest.shape[1]))

    targetTrain = tf.reshape(targetTrain, (numSamplesTrain // timeSteps, timeSteps, targetTrain.shape[1]))
    targetTest = tf.reshape(targetTest, (numSamplesTest // timeSteps, timeSteps, targetTest.shape[1]))

    print("featureTrain shape after reshaping:", featureTrain.shape)
    print("featureTest shape after reshaping:", featureTest.shape)
    print("targetTrain shape after reshaping:", targetTrain.shape)
    print("targetTest shape after reshaping:", targetTest.shape)

    # Define the Sequential model
    model = models.Sequential()

    model.add(layers.LSTM(units = 64, input_shape = (timeSteps, numFeatures)))

    #need two output layers. one for binary and one for numerical values

    #output for binary 0 for price decrease, 1 for price increase

    model.add(layers.Dense(1, activation = 'sigmoid' , name = 'binaryOutput'))

    #output for next day price

    model.add(layers.Dense(1, activation = 'linear', name = 'priceOutput'))


    model.compile(
        optimizer=Adam(),
        loss={
            'binaryOutput': 'binary_crossentropy',
            'priceOutput': 'mean_squared_error'
        },
        metrics={
            'binaryOutput': 'accuracy',
            'priceOutput': 'mae'
        }
    )

    model.summary()

    #print("featureTrain shape:", featureTrain.shape)
    #print("targetTrain shape:", targetTrain.shape)
    #print("featureTest shape:", featureTest.shape)
    #print("targetTest shape:", targetTest.shape)

    # Visualize the model
    #plot_model(model, to_file='newModelStructure.png', show_shapes=True, show_layer_names=True)

    binaryTargetTrain = targetTrain[:, -1, 0]
    priceTargetTrain = targetTrain[:, -1, 1]

    binaryTargetTest = targetTest[:, -1, 0]
    priceTargetTest = targetTest[:, -1, 1]

    model.fit(
    featureTrain,
    {
        'binaryOutput': binaryTargetTrain,  # Binary classification target
        'priceOutput': priceTargetTrain    # Price prediction target
    },
    epochs=10, #number of iterations the model will train for
    batch_size=32, #number of samples the model will use at once

    #validation data is for model performance
    validation_data=(
        featureTest,
        {
            'binaryOutput': binaryTargetTest,  # Binary classification target
            'priceOutput': priceTargetTest    # Price prediction target
        }
    )
)

if __name__ == "__main__":
    main()