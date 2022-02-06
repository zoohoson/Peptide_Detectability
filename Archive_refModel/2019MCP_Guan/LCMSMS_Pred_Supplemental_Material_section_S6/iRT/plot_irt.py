##Prediction of LCMSMS properties of peptides from sequence by deep learning
##Shenheng Guan1, Michael F. Moran, and Bin Ma
##2019-02-21

import pickle
import pandas as pd

fn = 'irt_reg_data_filtered_bidirLSTM2_masking_result.pickle'

with open(fn, 'rb') as fid:
    pred_data = pickle.load(fid)

print(pred_data.keys())

fn = 'irt_reg_data_filtered.pickle'
with open(fn, 'rb') as fid:
    data = pickle.load(fid)
    
print(data.keys())

import matplotlib.pyplot as plt

scale_factor = 30.

training_y = data['training_y']
predicted_training_y = pred_data['predicted_training_y']

testing_y = data['testing_y']
predicted_test_y = pred_data['predicted_test_y']


predicted_training_y *= scale_factor
predicted_test_y *= scale_factor

train_errors = predicted_training_y - training_y.reshape(training_y.shape[0],1)
test_errors = predicted_test_y - testing_y.reshape(testing_y.shape[0],1)

import numpy as np
bins = np.linspace(-10, 10, 101)
train_hist, bin_edges = np.histogram(train_errors, bins=bins)
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

df_train_hist = pd.DataFrame({'bin_centers': bin_centers,
                              'train_hist': train_hist})
df_train_hist.to_csv('train_hist.tsv', sep='\t', index=False)

test_hist, bin_edges = np.histogram(test_errors, bins=bins)
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

df_test_hist = pd.DataFrame({'bin_centers': bin_centers,
                              'test_hist': test_hist})
df_test_hist.to_csv('test_hist.tsv', sep='\t', index=False)

from scipy.optimize import curve_fit

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gaus(x, y):

    n=sum(y)
    mean = sum(x*y)/n                   
    sigma=np.sqrt(sum(y*(x-mean)**2)/n)

    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
    return popt, pcov

train_popt, train_pcov = fit_gaus(bin_centers, train_hist)
print('train_popt', train_popt)
plt.plot(bin_centers, train_hist)
plt.plot(bin_centers, gaus(bin_centers,*train_popt))
plt.title('train error')
plt.xlabel('Predicted iRT - Measured iRT')
plt.show()

plt.plot(training_y, predicted_training_y, '.', markersize = 1)
plt.title('train')
plt.xlim([-70, 170])
plt.ylim([-70, 170])
plt.xlabel('Measured iRT')
plt.ylabel('Predicted iRT')
plt.show()

test_popt, test_pcov = fit_gaus(bin_centers, test_hist)
print('test_popt', test_popt)
plt.plot(bin_centers, test_hist)
plt.plot(bin_centers, gaus(bin_centers,*test_popt))
plt.title('test error')
plt.xlabel('Predicted iRT - Measured iRT')
plt.show() 

plt.plot(testing_y, predicted_test_y, '.', markersize = 1)
plt.title('test')
plt.xlim([-70, 170])
plt.ylim([-70, 170])
plt.xlabel('Measured iRT')
plt.ylabel('Predicted iRT')
plt.show()
