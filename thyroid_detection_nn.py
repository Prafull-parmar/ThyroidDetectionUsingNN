import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

"""
Thyroid_detection using 
Multiclass MultiLayer Neural Network
"""

# Importing the dataset
dataset = pd.read_csv('Ann_train.csv')
X = dataset.iloc[:, 0:21].values
y = dataset.iloc[:, 21].values
y = pd.get_dummies(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Separate training and testing files
X_train, y_train = X, y
dataset = pd.read_csv('Ann_test.csv')
X_test = dataset.iloc[:, 0:21].values
y_test = dataset.iloc[:, 21].values
#y_test = pd.get_dummies(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to create model, required for KerasClassifier
def create_model():
  # Initialising the ANN
  classifier = Sequential()

  # Adding the input layer and the first hidden layer
  classifier.add(Dense(units = 10, activation = 'relu', input_dim =21 ))

  #Adding the second hidden layer
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

  #Adding the third hidden layer
  classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

  # Adding the output layer
  classifier.add(Dense(units = 3, activation = 'softmax'))

  # Compiling the ANN
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
  return classifier

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]

epochs = [10, 50, 100,200]

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

dic = grid.best_params_
best_batch_size= dic['batch_size']
best_epochs = dic['epochs']
classifier = create_model()
# Fitting the ANN with the best hyperparameters  to the Training set
classifier.fit(X_train, y_train, batch_size= best_batch_size, epochs = best_epochs)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred =1* (y_pred > 0.5)

# Creating the pred_class array
pred_class = []
for row in y_pred:
  row = list(row)
  pred_class.append(row.index(max(row))+1)
  
pred_class = np.array(pred_class)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, pred_class)

#Printing the Confusion matrix and accuracy
print(cm)
from sklearn.metrics import accuracy_score
print("Accuracy of the Test Set:%.2f" % (accuracy_score(pred_class,y_test)*100))

unique_elements, counts_elements = np.unique(y_test, return_counts=True)
print("Frequency of  the 3 Classes in the array:")
print(np.asarray((unique_elements, counts_elements)))

#First test record
arr = np.ones((1,21))
record1= arr*[0.42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0061,0.022,0.074,0.1,0.07392]
expected1 = 1
pred1= classifier.predict(record1)
print(pred1)
pred1 =1*(pred1 > 0.5)
print(pred1)
pred1 = list(pred1)
pred_class1 = (pred1.index(max(pred1))+1)
print("Expected Output:{}\nPredicted Output:{}".format(expected1,pred_class1))

#Second test record
arr = np.ones((1,21))
record1= arr*[0.82,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0.001,0.02,0.04,1,0.062]
expected1 = 2
pred1= classifier.predict(record1)
print(pred1)
pred1 =1*(pred1 > 0.5)
print(pred1)
pred1 = list(pred1[0])
pred_class1 = (pred1.index(max(pred1))+1)
print("Expected Output:{}\nPredicted Output:{}".format(expected1,pred_class1))

#Third test record
arr = np.ones((1,21))
record1= arr*[0.82,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0.001,0.02,0.04,1,0.062]
expected1 = 3
pred1= classifier.predict(record1)
print(pred1)
pred1 =1*(pred1 > 0.5)
print(pred1)
pred1 = list(pred1[0])
pred_class1 = (pred1.index(max(pred1))+1)
print("Expected Output:{}\nPredicted Output:{}".format(expected1,pred_class1))