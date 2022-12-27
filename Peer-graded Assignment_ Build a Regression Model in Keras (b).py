#!/usr/bin/env python
# coding: utf-8

# ## <center> <font color ="purple"> Download and Clean Dataset<font>

# <strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:</strong>
# 
# <strong>1. Cement</strong>
# 
# <strong>2. Blast Furnace Slag</strong>
# 
# <strong>3. Fly Ash</strong>
# 
# <strong>4. Water</strong>
# 
# <strong>5. Superplasticizer</strong>
# 
# <strong>6. Coarse Aggregate</strong>
# 
# <strong>7. Fine Aggregate</strong>

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


concrete_data = pd.read_csv('concrete_data.csv')
concrete_data.head()


# #### Let's check how many data points we have.

# In[4]:


concrete_data.shape


# we have so many data to train so we should be careful about data overfitting

# ## <font color ="red">Data Preprocessing  <font>

# In[5]:


concrete_data.describe()


# In[6]:


concrete_data.isnull().sum()


# It looks like there is no misssing values

# ## <font color ="red"> Split data into predictors and target<font>

# In[20]:


concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# In[21]:


predictors.head()


# In[22]:


target.head()


# In[37]:


#Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# In[23]:


n_cols = predictors.shape[1] # number of predictors
n_cols


# ## <font color ="red"> Import Keras <font>

# In[25]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[27]:


#Define our model
#our model has one hidden layer with 10 neurons and a ReLU activation function. It uses the adam optimizer and the mean squared error as the loss function.
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(1))

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')


# Let's import scikit-learn in order to randomly split the data into a training and test sets

# In[28]:


from sklearn.model_selection import train_test_split


# Splitting the data into a training and test sets by holding 30% of the data for testing

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# ## <font color ="red"> Train and Test the Network<font>

# In[30]:


# fit the model .we will train the model for 50 epochs.
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[33]:


#we need to evaluate the model on the test data.
loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


#we need to compute the mean squared error between the predicted concrete strength and the actual concrete strength.
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[36]:


#Create a list of 50 mean squared errors and report mean and the standard deviation of the mean squared errors.
total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))

