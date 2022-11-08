from tensorflow import keras # for building Neural Networks
print('Tensorflow/Keras: %s' % keras.__version__) # print version
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Dense, SimpleRNN # for creating regular densely-connected NN layers and RNN layers

# Data manipulation
import pandas as pd # for data manipulation
print('pandas: %s' % pd.__version__) # print version
import numpy as np # for data manipulation
print('numpy: %s' % np.__version__) # print version
import math # to help with data reshaping of the data

# Sklearn
import sklearn # for model evaluation
print('sklearn: %s' % sklearn.__version__) # print version
from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import mean_squared_error # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler # for feature scaling

# Visualization
import plotly 
import plotly.express as px
import plotly.graph_objects as go
print('plotly: %s' % plotly.__version__) # print version

# Other utilities
import sys
import os

# Assign main directory to a variable
main_dir=os.path.dirname(sys.path[0])

# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv(main_dir+'/data/weatherAUS.csv', encoding='utf-8')

# Drop records where target MinTemp=NaN or MaxTemp=NaN
df=df[pd.isnull(df['MinTemp'])==False]
df=df[pd.isnull(df['MaxTemp'])==False]

# Median daily temperature (mid point between Daily Max and Daily Min)
df['MedTemp']=df[['MinTemp', 'MaxTemp']].median(axis=1)

# Show a snaphsot of data
df

# Select only Canberra 
dfCan=df[df['Location']=='Canberra'].copy()

# Plot daily median temperatures in Canberra
fig = go.Figure()
fig.add_trace(go.Scatter(x=dfCan['Date'], 
                         y=dfCan['MedTemp'],
                         mode='lines',
                         name='Median Temperature',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                        ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Date'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Degrees Celsius'
                )

# Set figure title
fig.update_layout(title=dict(text="Median Daily Temperatures in Canberra", 
                             font=dict(color='black')))

fig.show()

def prep_data(datain, time_step):
    # 1. y-array  
    # First, create an array with indices for y elements based on the chosen time_step
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
    # Create y array based on the above indices 
    y_tmp = datain[y_indices]
    
    # 2. X-array  
    # We want to have the same number of rows for X as we do for y
    rows_X = len(y_tmp)
    # Since the last element in y_tmp may not be the last element of the datain, 
    # let's ensure that X array stops with the last y
    X_tmp = datain[range(time_step*rows_X)]
    # Now take this array and reshape it into the desired shape
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, 1))
    return X_tmp, y_tmp

X=dfCan[['MedTemp']]
scaler = MinMaxScaler()
X_scaled=scaler.fit_transform(X)


##### Step 2 - Create training and testing samples
train_data, test_data = train_test_split(X_scaled, test_size=0.2, shuffle=False)


##### Step 3 - Prepare input X and target y arrays using previously defined function
time_step = 7
X_train, y_train = prep_data(train_data, time_step)
X_test, y_test = prep_data(test_data, time_step)


##### Step 4 - Specify the structure of a Neural Network
model = Sequential(name="First-RNN-Model") # Model
model.add(Input(shape=(time_step,1), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
model.add(SimpleRNN(units=1, activation='tanh', name='Hidden-Recurrent-Layer')) # Hidden Recurrent Layer, Tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
model.add(Dense(units=1, activation='tanh', name='Hidden-Layer')) # Hidden Layer, Tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))
model.add(Dense(units=1, activation='linear', name='Output-Layer')) # Output Layer, Linear(x) = x


##### Step 5 - Compile keras model
model.compile(optimizer='adam', # default='rmsprop', an algorithm to be used in backpropagation
              loss='mean_squared_error', # Loss function to be optimized. A string (name of loss function), or a tf.keras.losses.Loss instance.
              metrics=['MeanSquaredError', 'MeanAbsoluteError'], # List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. 
              loss_weights=None, # default=None, Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs.
              weighted_metrics=None, # default=None, List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
              run_eagerly=None, # Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function.
              steps_per_execution=None # Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead.
             )


##### Step 6 - Fit keras model on the dataset
model.fit(X_train, # input data
          y_train, # target data
          batch_size=1, # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=20, # default=1, Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided
          verbose='auto', # default='auto', ('auto', 0, 1, or 2). Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. 'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
          callbacks=None, # default=None, list of callbacks to apply during training. See tf.keras.callbacks
          validation_split=0.0, # default=0.0, Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
          #validation_data=(X_test, y_test), # default=None, Data on which to evaluate the loss and any model metrics at the end of each epoch. 
          shuffle=True, # default=True, Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
          class_weight=None, # default=None, Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
          sample_weight=None, # default=None, Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only).
          initial_epoch=0, # Integer, default=0, Epoch at which to start training (useful for resuming a previous training run).
          steps_per_epoch=None, # Integer or None, default=None, Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined. 
          validation_steps=None, # Only relevant if validation_data is provided and is a tf.data dataset. Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch.
          validation_batch_size=None, # Integer or None, default=None, Number of samples per validation batch. If unspecified, will default to batch_size.
          validation_freq=1, # default=1, Only relevant if validation data is provided. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs.
          max_queue_size=10, # default=10, Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
          workers=1, # default=1, Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.
          use_multiprocessing=False, # default=False, Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. 
         )

# Predict the result on training data
pred_train = model.predict(X_train)
# Predict the result on test data
pred_test = model.predict(X_test)


##### Step 8 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
print("Note, the last parameter in each layer is bias while the rest are weights")
print("")
for layer in model.layers:
    print(layer.name)
    for item in layer.get_weights():
        print("  ", item)
print("")
print('---------- Evaluation on Training Data ----------')
print("MSE: ", mean_squared_error(y_train, pred_train))
print("")

print('---------- Evaluation on Test Data ----------')
print("MSE: ", mean_squared_error(y_test, pred_test))
print("")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(range(0,len(y_test))),
                         y=scaler.inverse_transform(y_test).flatten(),
                         mode='lines',
                         name='Median Temperature - Actual (Test)',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                        ))
fig.add_trace(go.Scatter(x=np.array(range(0,len(pred_test))),
                         y=scaler.inverse_transform(pred_test).flatten(),
                         mode='lines',
                         name='Median Temperature - Predicted (Test)',
                         opacity=0.8,
                         line=dict(color='red', width=1)
                        ))


# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Observation'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Degrees Celsius'
                )

# Set figure title
fig.update_layout(title=dict(text="Median Daily Temperatures in Canberra", 
                             font=dict(color='black')),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                 )

fig.show()

# With the current setup, we feed in 7 days worth of data and get the prediction for the next day
# We want to create an array that contains 7-day chunks offset by one day at a time
# This is so we can make a prediction for every day in the data instead of every 7th day
X_every=dfCan[['MedTemp']]
X_every=scaler.transform(X_every)
#X_every

for i in range(0, len(X_every)-time_step):
    if i==0:
        X_comb=X_every[i:i+time_step]
    else: 
        X_comb=np.append(X_comb, X_every[i:i+time_step])
X_comb=np.reshape(X_comb, (math.floor(len(X_comb)/time_step), time_step, 1))

print(X_comb.shape)

# Use the reshaped data to make predictions and add back into the dataframe
#np.zeros(time_step) - Set the first 7 numbers to 0 as we do not have data to predict
dfCan['MedTemp_prediction'] = np.append(np.zeros(time_step), scaler.inverse_transform(model.predict(X_comb)))
dfCan

fig = go.Figure()
fig.add_trace(go.Scatter(x=dfCan['Date'],
                         y=dfCan['MedTemp'],
                         mode='lines',
                         name='Median Temperature - Actual',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                        ))
fig.add_trace(go.Scatter(x=dfCan['Date'],
                         y=dfCan['MedTemp_prediction'],
                         mode='lines',
                         name='Median Temperature - Predicted',
                         opacity=0.8,
                         line=dict(color='red', width=1)
                        ))


# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Observation'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Degrees Celsius'
                )

# Set figure title
fig.update_layout(title=dict(text="Median Daily Temperatures in Canberra", 
                             font=dict(color='black')),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                 )

fig.show()

# Let's take the last sequence in the data to start predictions
inputs=X_comb[-1:]

# Create empty list
pred_list = []

# Loop 365 times to create predictions for the next year
for i in range(365):   
    pred_list.append(list(model.predict(inputs)[0])) # Generate prediction and add it to the list
    inputs = np.append(inputs[:,1:,:],[[pred_list[i]]],axis=1) # Drop oldest and append latest prediction

# Create a dataframe containing 365 days starting 2017-06-26
newdf=pd.DataFrame(pd.date_range(start='2017-06-26', periods=365, freq='D'), columns=['Date'])

# Add 365 days of model prediction from previous step
newdf['MedTemp_prediction']=scaler.inverse_transform(pred_list)

# Concatenate the orginal dataframe containing Canberra data and the new one containing predictions for the next 365 days
dfCan2=pd.concat([dfCan, newdf], ignore_index=False, axis=0, sort=False)
#dfCan2

fig = go.Figure()
fig.add_trace(go.Scatter(x=dfCan2['Date'][-730:],
                         y=dfCan2['MedTemp'][-730:],
                         mode='lines',
                         #name='Median Temperature (5-Day Moving Average)- Actual',
                         name='Median Temperature - Actual',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                        ))
fig.add_trace(go.Scatter(x=dfCan2['Date'][-730:],
                         y=dfCan2['MedTemp_prediction'][-730:],
                         mode='lines',
                         #name='Median Temperature (5-Day Moving Average) - Predicted',
                         name='Median Temperature - Predicted',
                         opacity=0.8,
                         line=dict(color='red', width=1)
                        ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Observation'
                )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black',
                 title='Degrees Celsius'
                )

# Set figure title
fig.update_layout(title=dict(text="Median Daily Temperatures in Canberra", 
                             font=dict(color='black')),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                 )

fig.show()