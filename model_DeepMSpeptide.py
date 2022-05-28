import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
tf.enable_eager_execution()


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def load_pep_and_codify(file, max_len):
    aa_dict={'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,
        'P':15,'O':16,'S':17,'U':18,'T':19,'W':20,'Y':21,'V':22}
##############################################
# original code
#     with open(file, 'r') as inf:
#         lines = inf.read().splitlines()
##############################################
    lines = file
    pep_codes=[]
    long_pep_counter = 0
    newLines = []
    for pep in lines:
        if not len(pep) > max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(current_pep)
            newLines.extend([pep])
        else:
            long_pep_counter += 1
    predict_data = keras.preprocessing.sequence.pad_sequences(pep_codes, value=0, padding='post', maxlen=max_len)
    return predict_data, long_pep_counter, newLines


def build_model():
    print('Loading model...')
    model_2_1D = keras.models.load_model('Model_original_DeepMSpeptide.h5')

    print('Initialize model...')
    model_2_1D.get_layer(index=0).set_weights(
    [tf.keras.initializers.glorot_uniform(seed=7)(shape=(23,50)).numpy()])
    model_2_1D.get_layer(index=1).set_weights([])
    model_2_1D.get_layer(index=2).set_weights(
        [tf.keras.initializers.glorot_uniform(seed=7)(shape=(3,50,128)).numpy(),
        np.zeros(128,)])
    model_2_1D.get_layer(index=3).set_weights(
        [tf.keras.initializers.glorot_uniform(seed=7)(shape=(2,128,64)).numpy(),
        np.zeros(64,)])
    model_2_1D.get_layer(index=4).set_weights([])
    model_2_1D.get_layer(index=5).set_weights(
        [tf.keras.initializers.glorot_uniform(seed=7)(shape=(64,64)).numpy(),
        np.zeros(64,)])
    model_2_1D.get_layer(index=6).set_weights([])
    model_2_1D.get_layer(index=7).set_weights([])
    model_2_1D.get_layer(index=8).set_weights(
        [tf.keras.initializers.glorot_uniform(seed=7)(shape=(64,1)).numpy(),
        np.zeros(1,)])
    model_2_1D.get_layer(index=9).set_weights([])
    return model_2_1D


def main():
    # train test split
    print('Loading input peptides')
    train = pd.read_csv('data/train_diff_hpp.csv')[['peptide', 'label']].drop_duplicates()
    test = pd.read_csv('data/test_diff_hpp.csv')[['peptide', 'label']].drop_duplicates()
    train, val = train_test_split(train, test_size=0.2, random_state=7)
    
    train_file = train.peptide.values
    train_data, train_skipped,  train_lines = load_pep_and_codify(train_file, 81)
    y_train = train.label.values
    print('Succesfully loaded {0} peptides and skipped {1}'.format(len(train_lines), str(train_skipped)))
    
    val_file = val.peptide.values
    val_data, val_skipped,  val_lines = load_pep_and_codify(val_file, 81)
    y_val = val.label.values
    print('Succesfully loaded {0} peptides and skipped {1}'.format(len(val_lines), str(val_skipped)))

    model_2_1D = build_model()
    history = model_2_1D.fit(train_data, y_train, epochs=200, 
                    batch_size=100,
                    validation_data=(val_data, y_val))

    np.save('log/DeepMSpeptide_log.npy', history.history)
    plt.figure(figsize=(16,2))
    plt.subplot(1,2,1)
    plot_graphs(history, 'acc')
    plt.subplot(1,2,2)
    plot_graphs(history, 'loss')
    plt.savefig('log/DeepMSpeptide.png', bbox_inches='tight')
    
    model_2_1D.save("log/DeepMspeptide.h5")


if __name__ == '__main__':
    main()