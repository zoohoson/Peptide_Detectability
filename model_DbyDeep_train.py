import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_npy_DbyDeep(df):
    label_enc = {v:k for k, v in enumerate('ZARNDCQEGHILKMFPSTWYV')}  # Z : 0
    pep_data = [[label_enc[aa] for aa in seq] + [0]*(81-len(seq))  # zero padding
               for seq in df.peptide.values]
    nterm_data = [[label_enc[aa] for aa in seq]
               for seq in df.nterm.values]
    cterm_data = [[label_enc[aa] for aa in seq]
               for seq in df.cterm.values]
    miss1_data = [[label_enc[aa] for aa in seq]
               for seq in df.miss1.values]
    miss2_data = [[label_enc[aa] for aa in seq]
               for seq in df.miss2.values]
    return np.array(pep_data), np.array(nterm_data), np.array(cterm_data), np.array(miss1_data), np.array(miss2_data), np.array(df.label.values)


def build_model():
    # Peptide Sequence Representation Module
    pep = tf.keras.layers.Input(shape=((81,)))
    pep_embed = tf.keras.layers.Embedding(21, 32, input_length=81, mask_zero=True)(pep)
    pep_embed = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(pep_embed)

    pep_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(pep_embed)
    pep_lstm = tf.keras.layers.Dense(16, activation='relu')(pep_lstm)
    pep_lstm = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(pep_lstm)

    # Cleavage Sites Representation Module
    n = tf.keras.layers.Input(shape=((15,)))
    n_embed = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)(n)
    n_embed = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(n_embed)
    c = tf.keras.layers.Input(shape=((15,)))
    c_embed = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)(c)
    c_embed = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(c_embed)
    m1 = tf.keras.layers.Input(shape=((15,)))
    m1_embed = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)(m1)
    m1_embed = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(m1_embed)
    m2 = tf.keras.layers.Input(shape=((15,)))
    m2_embed = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)(m2)
    m2_embed = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(m2_embed)

    n_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(n_embed)
    # n_lstm = tf.keras.layers.Dense(8, activation='relu')(n_lstm)
    n_lstm = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(n_lstm)
    c_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(c_embed)
    c_lstm = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(c_lstm)
    m1_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(m1_embed)
    m1_lstm = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(m1_lstm)
    m2_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(m2_embed)
    m2_lstm = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(m2_lstm)

    # Prediction Module
    merge = tf.keras.layers.concatenate([pep_lstm, 
                                         n_lstm,
                                         c_lstm,
                                         m1_lstm,
                                         m2_lstm])
    net_merge = tf.keras.layers.Dense(64, activation='relu')(merge)
    net_merge = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(net_merge)
    net_merge = tf.keras.layers.Dense(32, activation='relu')(net_merge)
    net_merge = tf.keras.layers.Dropout(np.random.uniform(0, 0.2))(net_merge)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(net_merge)

    model = tf.keras.Model(inputs=[pep, n, c, m1, m2],
                           outputs=[output])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(5e-4),
                  metrics=['accuracy'])
    return model


def to_log(log):
    with open("log/DbyDeep.log", "a+") as f:
        f.write(log+'\n')


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.title(metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def main():
    # gpu setting
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
        except RuntimeError as e:
            print(e)

    # train test split
    train = pd.read_csv('data/train_diff_hpp.csv')
    test = pd.read_csv('data/test_diff_hpp.csv')
    train, val = train_test_split(train, test_size=0.2, random_state=7)
    pep_train, n_train, c_train, m1_train, m2_train, label_train = get_npy_DbyDeep(train)
    pep_val, n_val, c_val, m1_val, m2_val, label_val = get_npy_DbyDeep(val)
    pep_test, n_test, c_test, m1_test, m2_test, label_test = get_npy_DbyDeep(test)
    print(pep_train.shape, n_train.shape, c_train.shape, m1_train.shape, m2_train.shape, label_train.shape)

    # train
    model = build_model()
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min', 
                                          verbose=1,
                                          patience=50)
    history = model.fit([pep_train, n_train, c_train, m1_train, m2_train], label_train,
                         epochs=200,
                         batch_size=256,

                         validation_data=([pep_val, n_val, c_val, m1_val, m2_val], label_val),
                         callbacks=[es],
                        )

    np.save('log/DbyDeep_log.npy', history.history)
    plt.figure(figsize=(16,2))
    plt.subplot(1,2,1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1,2,2)
    plot_graphs(history, 'loss')
    plt.savefig('log/DbyDeep.png', bbox_inches='tight')
    
    model.save("log/DbyDeep.h5")


if __name__ == '__main__':
    main()