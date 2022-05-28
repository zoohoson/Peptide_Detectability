from tensorflow.keras.models import model_from_json
#import keras.backend as K
import numpy as np
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#K.clear_session()
import time
import re

# coding
def trans(mer):
    l = len(mer)
    dic = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 
           'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 
           'V':17, 'W':18, 'Y':19, 'Z':20}
    a = [dic.get(mer[i]) for i in range(l)]
    return a

# coding sequences
def coding(mers):
    vectors = []
    for mer in mers:
        vectors.append(trans(mer))
    return vectors

# predicting
def predictor(data, resultpath, protease):
    print("Loading model...")
    s3 = time.time()
    # load architecture
    json_file = open('%s.json'%protease, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load parameters
    loaded_model.load_weights('%s.h5'%protease)
    e3 = time.time()
    print("Time cost of loading model was %s seconds."%(e3-s3))
    
    print("Predicting! Please wait...")
    s4 = time.time()
    # extract all the 31-mers
    mers = []
    for l in data:
        seqs = re.split('[\t||]', l[1])[1:]
        seqs1 = [j for j in seqs if j != '' and j != '*']
        mers.extend(seqs1)
    
    # predicting probabilities of all the sites
    X_test = np.array(coding(mers))
    probas = loaded_model.predict(X_test)
    proba = [p[0] for p in probas]
    
    # calculating the peptide probabilities
    i = 0
    results = []
    for line in data:
        array = line[1].split('\t')
        # Attention !! the predicted probability are for missing cut !!!
        if len(array) == 4:
            # left 31-mer's digestion probability
            if array[1] != '*':
                array[1] = float(1 - proba[i])
                i += 1
            else:
                array[1] = float(1)
            # right 31-mer digestion probability
            if array[2] != '*':
                array[2] = float(1 - proba[i])
                i += 1
            else:
                array[2] = float(1)

            # missing 31-mers probability
            mis_site = array[3].split('||')
            mispro = ''
            k = float(1)
            for lines in mis_site:
                if lines != '' and len(lines) == 31:
                    lines1 = proba[i]
                    i += 1
                    # line =trans(line)
                    k = k * lines1
                    lines1 = round(1 - lines1, 3)
                    mispro = mispro + str(lines1) + ','
                elif lines != '':
                    print(line)
                    print(lines)
            
            array[3] = mispro
            
            results.append([str(line[0]), str(array[0]), 
                            str(round(array[1],3)), str(round(array[2],3)), 
                            str(array[3])])
            
        elif len(array) == 1:
            results.append([str(line[0]), str(array[0]), 
                            '', '', ''])

    e4 = time.time()
    print("Time cost of prediction was %s seconds."%(e4-s4))
    
    s5 = time.time()
    with open(resultpath, 'w') as r:
        header = 'Protein_id' + '\t' + 'Peptide_sequence' + '\t' + \
                    'Digestibility_of_the_left_site' + '\t' + \
                    'Digestibility_of_the_right_site' + '\t' + \
                    'Digestibility_of_the_missed_site' + '\t' + '\n'
        r.write(header)
        for line in results:
            r.write('\t'.join(line) + '\n')
    
    e5 = time.time()
    print("Time cost of saving results was %s seconds."%(e5-s5))
