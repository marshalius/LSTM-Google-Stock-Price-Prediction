# Including Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

#Including Train Dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Scaling Training Set
scaler = MinMaxScaler()
training_set_scaled = scaler.fit_transform(training_set)

#Creating Timesteps in Training Set
X_train = []
y_train = []
for i in range(0, len(training_set_scaled)-60):
    X_train.append(training_set_scaled[i:i+60,0])
    y_train.append(training_set_scaled[i+60,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

#3D Reshaping Training Set
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building RNN
rnn = Sequential()
rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn.add(Dropout(rate=0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(rate=0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(rate=0.2))
rnn.add(LSTM(units=50))
rnn.add(Dropout(rate=0.2))
rnn.add(Dense(units=1))

rnn.compile(optimizer='adam', loss='mean_squared_error')

rnn.fit(X_train, y_train, epochs=100, batch_size = 32)

#Including Test Dataset
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real = dataset_test.iloc[:,1:2].values

#Concatenation Train and Test Datasets for Timesteps
dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
inputs = dataset_total.iloc[:,1:2].values
inputs_scaled = scaler.transform(inputs)

#Creating Timesteps in Test Set
X_test = []
for i in range(len(training_set), len(inputs)):
    X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)

#3D Reshaping Test Set
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Prediction With Test Set
pred = rnn.predict(X_test)
pred = scaler.inverse_transform(pred)

#comparison of Predicted Prices and Real Prices On the Plot
plt.plot(pred, color='red', label='Predicted Google Stock Prices')
plt.plot(real, color='blue', label='Real Google Stock Prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()