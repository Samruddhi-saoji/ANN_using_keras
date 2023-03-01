import numpy as np
import pandas as pd 


#import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
m = 2500 #dataset has 10k observations
X = dataset.iloc[:m, 3:-1].values #features
Y = dataset.iloc[:m, -1].values  #actual values



################ data pre-processing ###################

##### encoding the categorical columns #####
# Label encoding the column "Gender" 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
#encode col at index 2

#column "Geography"
#multiple possible classses => oneHot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    #encode the column at index 1
X = np.array(ct.fit_transform(X))
# column Geography --> columns France , Spain, Germany


##### split into training and test sets #####
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


##### feature scaling #####
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #no fitting on the test set to avoid info leakage



################### The ANN ############################
#import the necessary libraries/modules
from tensorflow import keras
from keras import layers, models

ann = models.Sequential()

#add the layers
ann.add(layers.Dense(units=6, activation='relu')) #1st hidden layer
ann.add(layers.Dense(units=6, activation='relu')) #2nd hidden layer
ann.add(layers.Dense(units=1, activation='sigmoid')) #output layer

#compile the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



############## Training and testing ####################
#train
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
    
#testing
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
    
#make a single prediction
input = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]  #input should be 2d
scaled_input = sc.transform(input)
ans = ann.predict(scaled_input) > 0.5
#print(ans)



############## Evaluating the ANN ####################
from sklearn.metrics import confusion_matrix, accuracy_score

#confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print("\nThe confusion matrix is:\n",matrix)

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
    #y_test = actual value
    #y_pred = predicted values
print("\nThe accuracy of the model is: ", accuracy)