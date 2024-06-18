import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


dataset = sklearn.datasets.load_breast_cancer()
print(dataset)

df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
print(df.head())

df['label'] = dataset.target

X = df.drop(columns='label', axis = 1)
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify = Y, random_state = 3)

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_test_std = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

model1 = LogisticRegression()
model2 = KNeighborsClassifier()
model3 = RandomForestClassifier()
model1.fit(X_train_std, Y_train)
model2.fit(X_train_std, Y_train)
model3.fit(X_train_std, Y_train)
test_data_prediction1 = model1.predict(X_test_std)
test_data_prediction2 = model2.predict(X_test_std)
test_data_prediction3 = model3.predict(X_test_std)
accuracy1 = accuracy_score(Y_test, test_data_prediction1)
accuracy2 = accuracy_score(Y_test, test_data_prediction2)
accuracy3 = accuracy_score(Y_test, test_data_prediction3)
print(f'The accuracy of LogisticRegression is: {accuracy1}')
print(f'The accuracy of KNeighborsClassifier is: {accuracy2}')
print(f'The accuracy of RandomForestClassifier is: {accuracy3}')

model = keras.Sequential([keras.layers.Flatten(input_shape = (30,)),
                          keras.layers.Dense(20, activation = 'relu'),
                          keras.layers.Dense(30, activation = 'relu'),
                          keras.layers.Dense(2, activation = 'sigmoid')])


optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = optimizer,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train_std, Y_train, validation_split = 0.1, epochs = 30)

loss, accuracy = model.evaluate(X_test_std, Y_test)
Y_pred = model.predict(X_test_std)
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)
input_data_as_numpy = np.asarray(input_data)
input_reshape = input_data_as_numpy.reshape(1,-1)
input_std = scaler.transform(input_reshape)
prediction = model.predict(input_std)
print(prediction)
prediction_label = [np.argmax(prediction)]
print(prediction_label)
if(prediction_label[0] == 0):
  print('The tumor is Malignant')
else:
  print('The tumor is Benign')

