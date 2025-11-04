# Deep-Learning-Exp5

DL-Implement a Recurrent Neural Network model for stock price prediction.

## **AIM**

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## **THEORY**

**Neural Network Model**

<img width="890" height="471" alt="image" src="https://github.com/user-attachments/assets/67fbb2eb-4d34-4f21-822a-b1329dd2f677" />


## **DESIGN STEPS**

STEP 1: Data Loading and Preprocessing

  - Load training and testing datasets using Pandas.

  - Extract the ‘Open’ column for analysis and apply MinMaxScaler to normalize data between 0 and 1


STEP 2: Training Data Preparation

  - Create input sequences of 60 previous days to predict the next day’s price.

  - Separate features (X_train) and targets (y_train) and reshape them into 3D format suitable for RNN input.

STEP 3: Model Construction

  - Build a Sequential RNN model with one SimpleRNN layer (40 units) and one Dense output layer (1 unit).

STEP 4: Model Compilation and Training

  - Compile the model using the Adam optimizer and Mean Squared Error (MSE) loss.

  - Train the model for 25 epochs with a batch size of 64.

STEP 5: Testing and Prediction

  - Combine train and test data, apply scaling, and prepare test sequences.

  - Predict stock prices using the trained model and convert them back to original scale using inverse transformation.

STEP 6: Visualization

  - Plot actual and predicted stock prices on a graph to visually compare performance and trend accuracy.

## **PROGRAM**


**Name:** YOGAVARMA B

**Register Number:** 2305002029


```python

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import layers, Sequential

# Load and scale data
train = pd.read_csv('trainset1.csv'); test = pd.read_csv('testset1.csv')
sc = MinMaxScaler((0,1))
train_scaled = sc.fit_transform(train.iloc[:,1:2])

# Generate training data
X_train = np.array([train_scaled[i-60:i,0] for i in range(60,len(train_scaled))])
y_train = np.array([train_scaled[i,0] for i in range(60,len(train_scaled))])
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build and train model
model = Sequential([layers.SimpleRNN(40,input_shape=(60,1)), layers.Dense(1)])
model.compile('adam','mse')
model.fit(X_train,y_train,epochs=25,batch_size=64,verbose=0)

# Prepare input and test data
inputs = pd.concat((train['Open'], test['Open']), axis=0).values.reshape(-1,1)
X_test = np.array([sc.transform(inputs)[i-60:i,0] for i in range(60,len(inputs))]).reshape(-1,60,1)

# Predict and visualize
pred = sc.inverse_transform(model.predict(X_test))
plt.plot(inputs,color='yellow',label='Real Price')
plt.plot(range(60,len(inputs)),pred,color='violet',label='Predicted')
plt.title('Google Stock Price Prediction'); plt.xlabel('Time'); plt.ylabel('Price'); plt.legend(); plt.show()

````





## **OUTPUT**


**True Stock Price, Predicted Stock Price vs time**

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/0a8df849-9b15-4f8d-a679-6e68a9b9bc32" />



## **RESULT**

Thus the stock price is predicted using Recurrent Neural Networks successfully.
