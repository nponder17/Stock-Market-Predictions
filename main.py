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

     print(correlationMatrix)

     return potentialFeatures


def getTensorFlowPredictions(allFeatures, prefix):


    # Create features for the model
    features = allFeatures.drop(columns=['Next Day Price']).values
    priceChange = allFeatures["Next Day Price"].values - allFeatures[prefix + '_Price'].values
    targetGainLoss = (priceChange > 0).astype(int)
    targetPrice = allFeatures["Next Day Price"].values

    targets = np.column_stack((targetGainLoss, targetPrice))

    print("targetGainLoss shape:", targetGainLoss.shape)
    print("targetPrice shape:", targetPrice.shape)

    # Convert features and target into tensor flow tensors
    featureTensor = tf.convert_to_tensor(features, dtype=tf.float32)
    targetTensor = tf.convert_to_tensor(targets, dtype=tf.float32)

    # Split data
    trainingSize = int(0.8 * len(featureTensor))
    featureTrain, featureTest = featureTensor[:trainingSize], featureTensor[trainingSize:]
    targetTrain, targetTest = targetTensor[:trainingSize], targetTensor[trainingSize:]

    # Normalize the feature data
    scaler = StandardScaler()
    featureTrain = scaler.fit_transform(featureTrain)  # Fit on train, transform train
    featureTest = scaler.transform(featureTest)  # Only transform test

    # Scale the target data
    targetScaler = StandardScaler()
    targetTrain_numpy = targetTrain.numpy()  # Convert to NumPy array
    targetTest_numpy = targetTest.numpy()  # Convert to NumPy array

    # Reshaping the target arrays using NumPy
    targetTrain_numpy[:, 1] = targetScaler.fit_transform(targetTrain_numpy[:, 1].reshape(-1, 1)).flatten()  # Scaling only price
    targetTest_numpy[:, 1] = targetScaler.transform(targetTest_numpy[:, 1].reshape(-1, 1)).flatten()  # Scaling only price

    # Convert them back to tensors
    targetTrain = tf.convert_to_tensor(targetTrain_numpy, dtype=tf.float32)
    targetTest = tf.convert_to_tensor(targetTest_numpy, dtype=tf.float32)

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

    # Extract the price targets for training and testing
    priceTargetTrain = targetTrain[:, -1, 1]  # Price target for train
    priceTargetTest = targetTest[:, -1, 1]    # Price target for test

    input_layer = layers.Input(shape=(timeSteps, numFeatures))

    lstm_1 = layers.LSTM(64, return_sequences=True)(input_layer)
    lstm_2 = layers.LSTM(32)(lstm_1)

    # Price Prediction Output (Regression)
    price_output = layers.Dense(1, activation='linear', name='priceOutput')(lstm_2)

    # Gain/Loss Prediction Output (Classification)
    classification_output = layers.Dense(1, activation='sigmoid', name='classOutput')(lstm_2)

    # Define model with two outputs
    model = models.Model(inputs=input_layer, outputs=[price_output, classification_output])

    model.compile(
        optimizer=Adam(),
        loss={
            'priceOutput': 'mean_squared_error',
            'classOutput': 'binary_crossentropy'
        },
        metrics={
            'priceOutput': ['mae'],
            'classOutput': ['accuracy']
        }
    )

    model.summary()

    # Targets
    priceTargetTrain = targetTrain[:, -1, 1]
    priceTargetTest = targetTest[:, -1, 1]
    classTargetTrain = targetTrain[:, -1, 0]
    classTargetTest = targetTest[:, -1, 0]

    # Fit the model
    model.fit(
        featureTrain,
        {'priceOutput': priceTargetTrain, 'classOutput': classTargetTrain},
        epochs=20,
        batch_size=32,
        validation_data=(featureTest, {
            'priceOutput': priceTargetTest,
            'classOutput': classTargetTest
        })
    )

    # Evaluate
    test_metrics = model.evaluate(
        featureTest,
        {'priceOutput': priceTargetTest, 'classOutput': classTargetTest}
    )


    # Predict
    price_preds, class_preds = model.predict(featureTest)

    # Convert classification predictions to 0 or 1
    class_preds_binary = (class_preds > 0.5).astype(int)

    # Plot gain/loss
    plt.figure(figsize=(8, 4))
    plt.plot(classTargetTest, label='Actual Gain/Loss', linestyle='--')
    plt.plot(class_preds_binary, label='Predicted Gain/Loss', alpha=0.7)
    plt.title('Gain/Loss Prediction')
    plt.legend()
    plt.show()

        # Plot the actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(priceTargetTest, label="Actual Prices", color='blue')
    plt.plot(price_preds, label="Predicted Prices", color='red')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Sample Index")
    plt.ylabel("Price (scaled)")
    plt.legend()
    plt.show()

    #graph using validation data

        # Graph using validation data
    val_price_preds, val_class_preds = model.predict(featureTest)
    val_class_preds_binary = (val_class_preds > 0.5).astype(int)

    # Plot the actual vs predicted prices using validation data
    plt.figure(figsize=(10, 6))
    plt.plot(priceTargetTest, label="Actual Prices (Validation)", color='blue')
    plt.plot(val_price_preds, label="Predicted Prices (Validation)", color='red')
    plt.title("Validation: Actual vs Predicted Prices")
    plt.xlabel("Sample Index")
    plt.ylabel("Price (scaled)")
    plt.legend()
    plt.show()

    # Plot validation gain/loss predictions
    plt.figure(figsize=(8, 4))
    plt.plot(classTargetTest, label='Actual Gain/Loss (Validation)', linestyle='--')
    plt.plot(val_class_preds_binary, label='Predicted Gain/Loss (Validation)', alpha=0.7)
    plt.title('Validation: Gain/Loss Prediction')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()