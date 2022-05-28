import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import json
import joblib
from mrmr import mrmr_classif


def labelling_ap3(prot2cnt, prot2cnt_miss, THRESHOLD = 4):  # prot2cnt, prot2cnt_miss must be list split by ;

    ts2label = dict()
    # protein loop
    for idx in range(len(prot2cnt)):
        # amino acid loop in one protein
        for amino_idx in range(len(prot2cnt[idx])):
            # tryptic site : K, R
            if prot2cnt[idx][amino_idx][:1] in 'KR':
                
                # *******TRYPTIC SITE EXTRACTING FROM PROTEIN*******
                # tryptic site in protein N-terminal
                if amino_idx <= THRESHOLD - 1:
                    tryptic_site = 'Z'*(THRESHOLD-amino_idx) + ''.join([a[0] for a in prot2cnt[idx][: amino_idx+THRESHOLD + 1]])  # added Z
                # tryptic site in protein C-terminal
                elif amino_idx >= len(prot2cnt[idx]) - THRESHOLD:
                    tryptic_site = ''.join([a[0] for a in prot2cnt[idx][amino_idx-THRESHOLD: amino_idx+THRESHOLD + 1]]) + 'Z'*(THRESHOLD-(len(prot2cnt[idx])-1)+amino_idx)  # added Z
                else:
                    tryptic_site = ''.join([a[0] for a in prot2cnt[idx][amino_idx-THRESHOLD: amino_idx+THRESHOLD + 1]])

                    
                # *******TRYPTIC SITE LABELLING (WHEN POSITIVE)*******
                # N-terminal of tryptic site condition
                N_Pcondition = int(prot2cnt[idx][amino_idx][2:]) >= 1
                # C-terminal of tryptic site condition
                if amino_idx == len(prot2cnt[idx]) - 1:  # tryptic site in last position of protein
                    C_prot2cntondition = True
                else:
                    C_prot2cntondition = int(prot2cnt[idx][amino_idx + 1][2:]) >= 1
                # miss cleavage of tryptic site contidion
                miss_prot2cntondition = int(prot2cnt_miss[idx][amino_idx][2:]) == 0
                # True of all above, Positive labeling
                P_condition = (N_Pcondition == True) and (C_prot2cntondition == True) and (miss_prot2cntondition == True)

                # *******TRYPTIC SITE LABELLING (WHEN NEGATIVE)*******
                # N-terminal of tryptic site condition
                N_NCondition = int(prot2cnt[idx][amino_idx][2:]) == 0
                # C-terminal of tryptic site condition
                if amino_idx == len(prot2cnt[idx]) - 1:  # tryptic site in last position of protein
                    C_NCondition = False
                else:
                    C_NCondition = int(prot2cnt[idx][amino_idx + 1][2:]) == 0
                # miss cleavage of tryptic site condition
                miss_Ncondition = int(prot2cnt_miss[idx][amino_idx][2:]) >= 1
                # True of all above, Negative labeling
                N_condition = (N_NCondition == True) and (C_NCondition == True) and (miss_Ncondition == True)

                # *******MAKING DATASET*******
                if P_condition:
                    if tryptic_site not in ts2label:
                        ts2label[tryptic_site] = []
                    ts2label[tryptic_site].append(('P', idx, amino_idx))
                elif N_condition:
                    if tryptic_site not in ts2label:
                        ts2label[tryptic_site] = []
                    ts2label[tryptic_site].append(('N', idx, amino_idx))
    
    # *******JUST FOR COUNTING*******
    p = [k for k, v in ts2label.items() if v[0][0]=='P']
    n = [k for k, v in ts2label.items() if v[0][0]=='N']
    print(' P, unique p: {} \n n, unique n: {} \n total : {:,}'.format(
         (len(p), len(set(p))), 
         (len(n), len(set(n))), 
         len(p)+len(n)))
    df = pd.DataFrame({'tryptic_site':p+n, 'label':['P' if i in p else 'N' for i in p+n]})
    return df, ts2label


def onehot_enc(aa):
    return [1 if aa == a else 0 for a in 'ARNDCQEGHILKMFPSTWYVZ']

def labelling_detect(df, aa2val, RF_digest):
    df_ = df.copy()
    # set tryptic site 9mer
    df_['PEP'] = df_['peptide'].values
    df_['En'] = df_['nterm'].values
    df_['Ec'] = df_['cterm'].values
    df_['E1'] = df_['miss1'].values
    df_['E2'] = df_['miss2'].values
    df_['miss'] = [sum([1 for _ in p[:-1] if _ in 'KR']) for p in df_.PEP.values]
    df_['En'] = df_.En.apply(lambda x: x[3:-3]).values
    df_['Ec'] = df_.Ec.apply(lambda x: x[3:-3]).values
    df_['E1'] = df_.E1.apply(lambda x: x[3:-3] if x != '-' else '-').values
    df_['E2'] = df_.E2.apply(lambda x: x[3:-3] if x != '-' else '-').values

    # digestibility
    score_n = cal_digestibility(df_.En.values, RF_digest)
    score_c = cal_digestibility(df_.Ec.values, RF_digest)
    score_m1 = cal_digestibility(df_.E1.values, RF_digest)
    score_m2 = cal_digestibility(df_.E2.values, RF_digest)
    df_['digestibility'] = score_n * score_c * score_m1 * score_m2
    
    # get AA index
    cols = list(range(21+len(aa2val['A'])))  # aa_cnt + aa_index length sum 
    cols_value = []
    for seq in df_.peptide.values:
        aa_cnt = [seq.count(a) for a in 'ARNDCQEGHILKMFPSTWYVZ']
        aa_index = list(np.array([aa2val[aa] for aa in seq]).sum(axis=0))
        cols_value.append(aa_cnt+aa_index)
    df_[cols] = cols_value

    df_.drop(['peptide', 'En', 'Ec', 'E1', 'E2', 'PEP', 'nterm', 'cterm', 'miss1', 'miss2'], axis=1, inplace=True)
    return df_


def cal_digestibility(seq_li, model):
    noseq_idx = [idx for idx, _ in enumerate(seq_li) if _ == '-']
    X = np.array([[__ for _ in seq for __ in onehot_enc(_)] if seq != '-' else [0]*189 for seq in seq_li])
    y_pred = model.predict_proba(X)[:, 1]  # positive probability = digestibility
    y_pred[noseq_idx] = 1
    return y_pred


def main():
    start_time = time.time()
    
    # preprocessing of massIVE-KB
    print('### loading digestibility data ...\t\t\t')
    f = open('data/data_digestibility_AP3.json')
    data = json.load(f)
    f.close()
    rows = []
    for pn, pi in data.items():
        protein = pn
        sequence = pi['sequence']
        spectral_cnt = ';'.join([aa+'_'+str(cnt) for aa, cnt in zip(sequence, pi['cleavage_cnt'])])
        miss_cleavage_cnt = ';'.join([aa+'_'+str(cnt) for aa, cnt in zip(sequence, pi['miss_cleavage_cnt'])])
        rows.append([protein, sequence, spectral_cnt, miss_cleavage_cnt])
    df_digest_protein = pd.DataFrame(rows, columns=['PROTEIN', 'SEQUENCE', 'SPECTRAL_CNT', 'SPECTRAL_CNT_MISS'])
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish loading digestibility data ... time elapsed {elapsed_time} min \t\t\t')

    prot2cnt = [_.split(';') for _ in df_digest_protein.SPECTRAL_CNT.values]
    prot2cnt_miss = [_.split(';') for _ in df_digest_protein.SPECTRAL_CNT_MISS.values]
    df_digest, ts2label = labelling_ap3(prot2cnt, prot2cnt_miss, THRESHOLD=4)  # ts2label is just for checking

    # training digestibility
    print('### training digestibility model ...\t\t\t', end='\r')
    X = np.array([[__ for _ in ts for __ in onehot_enc(_)] for ts in df_digest.tryptic_site.values])
    y = df_digest.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25, random_state=77)
    RF_digest = RandomForestClassifier(n_estimators=200, max_features='sqrt')
    RF_digest.fit(X, y)
    joblib.dump(RF_digest, 'log/AP3_digestibility.joblib')  # RF_digest = joblib.load("path/digestibility_AP3.joblib")
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish training digestibility model ... time elapsed {elapsed_time} min \t\t\t')
    
    # training detectability
    print('### training detectability model ...\t\t\t', end='\r')
    df_detect_peptide_train = pd.read_csv('data/train_diff_hpp.csv')
    df_detect_peptide_test = pd.read_csv('data/test_diff_hpp.csv')
    # AA index
    df_aaindex = pd.read_csv('data/aaindex/df_aaindex.csv')
    tmp = df_aaindex.drop('Unnamed: 0',axis=1).T
    aa2val = dict()
    for aa, val in zip(tmp.index, tmp.values):
        aa2val[aa]=val
    train = labelling_detect(df_detect_peptide_train, aa2val, RF_digest)
    X = train.drop('label', axis=1)
    y = train['label'].values
    # use mrmr classification
    selected_features = mrmr_classif(X, y, K = 50)
    #############################
    # start = time.time()
    # col2auc = dict()
    # for col_idx in range(1, len(selected_features)+1):
    #     print(col_idx, round(time.time() - start, 2), end='\r')
        
    #     cols = selected_features[:col_idx] + ['label']
    #     df_selection = train[cols]
    #     X = df_selection.drop('label', axis=1).values
    #     y = df_selection['label'].values
    #     clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=7)
    #     scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    #     col2auc[col_idx] = sum(scores) / len(scores)
    # for i in sorted(col2auc.items(), key=lambda x:x[1], reverse=True):
    #     print(i)  # select 29 featuresa
    #####################################

    df = pd.concat([df_detect_peptide_train, df_detect_peptide_test], axis=0)
    train_idx = df_detect_peptide_train.shape[0]
    df_ = labelling_detect(df, aa2val, RF_digest)
    train_final = df_.iloc[:train_idx]
    test_final = df_.iloc[train_idx:]
    cols = selected_features[:29] + ['label']
    train = train_final[cols]
    test = test_final[cols]
    test.to_csv('data/test_diff_HPP_AP3.csv', index=False)
    X_train = train_final.drop('label', axis=1).values
    y_train = train_final['label'].values
    X_test = test_final.drop('label', axis=1).values
    y_test = test_final['label'].values
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    RF_detect = RandomForestClassifier(n_estimators=200, max_features='sqrt')
    RF_detect.fit(X_train, y_train)
    joblib.dump(RF_detect, 'log/AP3_detectability.joblib')
    print(f'### finish training detectability model ... time elapsed {elapsed_time} min \t\t\t')

if __name__ == main():
    main()