##Prediction of LCMSMS properties of peptides from sequence by deep learning
##Shenheng Guan1, Michael F. Moran, and Bin Ma
##2019-02-21

import pickle
import numpy as np

data_type = 'data'
model_type = 'bidirLSTM2_masking'

scale_factor = 30.

data_fn = 'irt_reg_data_filtered' + '.pickle'
with open(data_fn, 'rb') as f:
  irt_reg_data = pickle.load(f)
print(irt_reg_data.keys())

training_X = irt_reg_data['training_X']
training_y = irt_reg_data['training_y']/scale_factor
testing_X = irt_reg_data['testing_X']
testing_y = irt_reg_data['testing_y']/scale_factor
print(training_X.shape, training_y.shape, testing_X.shape, testing_y.shape)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Masking(mask_value=0., input_shape=(None, training_X.shape[2])))
network.add(layers.Bidirectional(layers.LSTM(256,
			 		     dropout=0.1,
					     recurrent_dropout=0.5,  
					     return_sequences=True)))
network.add(layers.Bidirectional(layers.LSTM(256,
			 		     dropout=0.1,
					     recurrent_dropout=0.5)))
network.add(layers.Dense(256, activation='tanh'))
network.add(layers.Dense(32, activation='tanh'))
network.add(layers.Dense(1))
print(network.summary())

network.compile(optimizer='adam', loss='mean_squared_error')
history = network.fit(training_X, 
		      training_y, 
		      epochs=150, 
		      batch_size=64,
		      validation_split=0.2)

cost = network.evaluate(testing_X, testing_y, batch_size=64)
print('test cost:', cost)
from keras.models import load_model
network.save(data_fn.replace('.pickle', '_' + model_type + '_model.h5')) 

predicted_training_y = network.predict(training_X)
print(predicted_training_y.shape)
predicted_test_y = network.predict(testing_X)
print(predicted_test_y.shape)
training_errors = predicted_training_y.ravel() - training_y
testing_errors = predicted_test_y.ravel() - testing_y
print(predicted_training_y.shape, training_y.shape, predicted_test_y.shape, testing_y.shape)
irt_reg_result = {'predicted_training_y':predicted_training_y, 'predicted_test_y':predicted_test_y}
results_fn = data_fn.replace('.pickle', '_' + model_type + '_result.pickle')
print(results_fn)
with open(results_fn, 'wb') as handle:
    pickle.dump(irt_reg_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

from sklearn.metrics import r2_score
print('test', r2_score(testing_y, predicted_test_y))
print('train', r2_score(training_y, predicted_training_y))

