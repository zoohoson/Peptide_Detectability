import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 2. RTX A6000
# gpu setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    except RuntimeError as e:
        print(e)


def get_npy_DbyDeep(df):
    label_enc = {v:k for k, v in enumerate('ZARNDCQEGHILKMFPSTWYV')}  # Z : 0
    pep_data = [[label_enc[aa] for aa in seq] + [0]*(40-len(seq))  # zero padding
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


    # pep_lstm = tf.keras.layers.Dense(16, activation='relu')(pep_lstm)
    # mcleavage_site_dnn = tf.keras.layers.Dense(8, activation='relu')
    # net_merge = tf.keras.layers.Dense(32, activation='relu')(net_merge)
    # net_merge = drop_out(net_merge)

def build_model(ablation):
    # Peptide Sequence Representation Module
    peptide_embedding_matrix = tf.keras.layers.Embedding(21, 16, input_length=40, mask_zero=True)
    peptide_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))
    drop_out = tf.keras.layers.Dropout(np.random.uniform(0, 0.4))

    pep = tf.keras.layers.Input(shape=((40,)))
    pep_embed = peptide_embedding_matrix(pep)
    pep_lstm = peptide_rnn(pep_embed)
    pep_lstm = drop_out(pep_lstm)

    # Cleavage Sites Representation Module
    cleavage_site_embedding_marix = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)
    cleavage_site_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))
    mcleavage_site_embedding_marix = tf.keras.layers.Embedding(21, 16, input_length=15, mask_zero=True)
    mcleavage_site_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))
    # version03 : 64
    # version04 : 16
    n = tf.keras.layers.Input(shape=((15,)))
    n_embed = cleavage_site_embedding_marix(n)
    n_lstm = cleavage_site_rnn(n_embed)
    n_lstm = drop_out(n_lstm)
    c = tf.keras.layers.Input(shape=((15,)))
    c_embed = cleavage_site_embedding_marix(c)
    c_lstm = cleavage_site_rnn(c_embed)
    c_lstm = drop_out(c_lstm)
    m1 = tf.keras.layers.Input(shape=((15,)))
    m1_embed = mcleavage_site_embedding_marix(m1)
    m1_lstm = mcleavage_site_rnn(m1_embed)
    m1_lstm = drop_out(m1_lstm)
    m2 = tf.keras.layers.Input(shape=((15,)))
    m2_embed = mcleavage_site_embedding_marix(m2)
    m2_lstm = mcleavage_site_rnn(m2_embed)
    m2_lstm = drop_out(m2_lstm)

    # Prediction Module
    if ablation:
        net_merge = pep_lstm
    else:
        merge = tf.keras.layers.concatenate([pep_lstm, 
                                         n_lstm,
                                         c_lstm,
                                         m1_lstm,
                                         m2_lstm])
        net_merge = tf.keras.layers.Dense(64, activation='relu')(merge)

    net_merge = drop_out(net_merge)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(net_merge)

    if ablation:
        model = tf.keras.Model(inputs=[pep], outputs=[output])
    else:
        model = tf.keras.Model(inputs=[pep, n, c, m1, m2], outputs=[output])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])
    return model


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.title(metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def main(train_file_path, version, ablation):
    # train test split
    train = pd.read_csv(train_file_path)
    train, val = train_test_split(train, test_size=0.2, random_state=7)
    pep_train, n_train, c_train, m1_train, m2_train, label_train = get_npy_DbyDeep(train)
    pep_val, n_val, c_val, m1_val, m2_val, label_val = get_npy_DbyDeep(val)
    print(pep_train.shape, n_train.shape, c_train.shape, m1_train.shape, m2_train.shape, label_train.shape)

    # train
    model = build_model(ablation)
    print(model.summary())
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          mode='min', 
                                          verbose=1,
                                          patience=50)
    cp = tf.keras.callbacks.ModelCheckpoint(
        f'log/model_DbyDeep_{version}_{ablation}.h5',
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False, 
        mode='auto', 
        period=1)
    if ablation:
        history = model.fit(pep_train, label_train,
                    epochs=200,
                    batch_size=256,

                    validation_data=(pep_val, label_val),
                    callbacks=[es, cp],
                    )
    else:
        history = model.fit([pep_train, n_train, c_train, m1_train, m2_train], label_train,
                            epochs=200,
                            batch_size=256,

                            validation_data=([pep_val, n_val, c_val, m1_val, m2_val], label_val),
                            callbacks=[es, cp],
                            )
    np.save(f'log/log_DbyDeep_history_{version}_{ablation}.npy', history.history)
    plt.figure(figsize=(16,2))
    plt.subplot(1,2,1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1,2,2)
    plot_graphs(history, 'loss')
    plt.savefig(f'log/log_DbyDeep_plot_{version}_{ablation}.png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='/data/2021_SJH_detectability/data_human/train.csv', help='train file path')
    parser.add_argument('--version', type=str, help='version', required=True)
    parser.add_argument('--ablation', action='store_true', help='ablation flag')
    opt = parser.parse_args()
    print(opt)

    main(opt.train_file, opt.version, opt.ablation)