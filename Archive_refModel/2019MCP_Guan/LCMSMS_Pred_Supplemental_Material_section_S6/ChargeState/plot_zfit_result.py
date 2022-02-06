##Prediction of LCMSMS properties of peptides from sequence by deep learning
##Shenheng Guan1, Michael F. Moran, and Bin Ma
##2019-02-21

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

AA_vocabulary = 'KRPTNAQVSGILCMJHFYWEDBXOU'#B: acetyl; J: oxidized Met; X:pyro_glu

def one_hot_to_single_ptm(one_hot):
    seq = ''
    for row in one_hot:
        if row.sum() > 0:
            idx = row.argmax()
            seq += AA_vocabulary[idx]
    return seq

single2psi_lookup = {'B': '(Acetyl)-',
                     'J': 'M(Oxidation)',
                     'X': '(Gln->pyro-Glu)Q',
                     'C': 'C(Carbamidomethyl)'}

def single_ptm_to_psi(single_ptm_seq):
    psi_seq = single_ptm_seq
    for code in single2psi_lookup:
        psi_seq = psi_seq.replace(code, single2psi_lookup[code])
    return psi_seq

fn1 = 'zfit_bidirLSTM2_masking_result.pickle'
with open(fn1, 'rb') as fid:
    pred_data = pickle.load(fid)

fn2 = 'zfit_one_hot.pickle'
with open(fn2, 'rb') as fid:
    real_data = pickle.load(fid)

def plot_zfit(train_X, train_y, predicted_train_y, n):

    X = train_X[n]
    s_seq = one_hot_to_single_ptm(X)
    psi_seq = single_ptm_to_psi(s_seq)
   
    y_exp = train_y[n]
    y_pred = predicted_train_y[n]
    pcc = pearsonr(y_exp, y_pred)

    fig, ax = plt.subplots()
    charges = range(1, len(y_exp) + 1)
    ax.stem(charges, y_exp, 'b', markerfmt=" ", label='Experimental')
    ax.stem(charges, -y_pred, 'g', markerfmt=" ", label='Predicted')
    plt.text(1.5, -0.35, 'pcc: %6.3f, %6.3g'%(pcc[0], pcc[1]))
    plt.xticks(charges)
    plt.xlabel('charge')
    plt.title(psi_seq)
    plt.show()

train_X = real_data['train_X']
train_y = real_data['train_y']
test_y = real_data['test_y']
test_X = real_data['test_X']
predicted_test_y = pred_data['predicted_test_y']
predicted_train_y = pred_data['predicted_training_y']

#look at training data
n = 91994
#n = 112763   
plot_zfit(train_X, train_y, predicted_train_y, n)

###look at test data
##n = 1516
##plot_zfit(test_X, test_y, predicted_test_y, n)
