##Prediction of LCMSMS properties of peptides from sequence by deep learning
##Shenheng Guan1, Michael F. Moran, and Bin Ma
##2019-02-21

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'AminoAcids-Peptides')
import pep_mass

simple_HCD_ion_types = ['b', 'y']

def get_frag_position(sequence, sequence_length, ion_type, ion_number, frag_charge):
    if ion_type == 'b':
        col_idx = ion_number - 1
        if ('(Acetyl)-' in sequence or '(Carbamyl)' in sequence) \
           and col_idx < sequence_length - 1:
            col_idx += 1
    if ion_type == 'y':
        col_idx = sequence_length - ion_number

    if ion_type == 'b':
        if frag_charge == 2:
            row_idx = 0 #b^2
        else:
            row_idx = 1 #b
            
    if ion_type == 'y':
        if frag_charge == 2:
            row_idx = 3 #y^2
        else:
            row_idx = 2 #y

    return (row_idx, col_idx)
             

def filter_ion_table(ion_table, bare_sequence, sequence, charge):
    filtered_table = []
    for ion in ion_table:
        if '-' not in ion[0] and \
           (ion[0][0] == 'b' or ion[0][0] == 'y'):
            if charge > 2:
                filtered_table.append(ion)
            else:
                if '^' not in ion[0]:
                    filtered_table.append(ion)

    no_ion_series = 4#b^2, b, y, and y^2 in order
    sequence_length = len(bare_sequence)
    ion_mzs = np.zeros((no_ion_series, sequence_length))
    ion_labels = np.zeros((no_ion_series, sequence_length)).astype(str)

    for ion in filtered_table:
        ion_type = ion[0][0]
        if '^' in ion[0]:
            stmp = ion[0].split('^')
            frag_charge = int(stmp[1])
            ion_number = int(stmp[0][1:])
        else:
            frag_charge = 1
            ion_number = int(ion[0][1:])

        if frag_charge <= 2:
            row_idx, col_idx = get_frag_position(sequence,
                                                 sequence_length,
                                                 ion_type,
                                                 ion_number,
                                                 frag_charge)
            
            ion_mzs[row_idx, col_idx] = ion[1]
            ion_labels[row_idx, col_idx] = ion[0] 
        
    ion_mzs = ion_mzs.T
    ion_labels = ion_labels.T
    if charge <= 2:
        ion_mzs = ion_mzs[:, 1:3]
        ion_labels = ion_labels[:, 1:3]
        
    return filtered_table, ion_mzs, ion_labels

def plot_spectrum(all_ions, type_tag, ion_index):
    test_ions = all_ions[type_tag]
    sequence =test_ions[ion_index]['seq']
    peptide_charge = test_ions[ion_index]['charge']
    exp_ions = test_ions[ion_index]['exp_ions']
    pred_ions = test_ions[ion_index]['pred_ions']
    pcc = test_ions[ion_index]['pcc']
                                                  
    whole,tmp_ion_table=pep_mass.makeSequenceIons(simple_HCD_ion_types,
                                                  sequence,
                                                  peptide_charge,
                                                   (50, 2000))

    peptide_mz = whole[-1]/peptide_charge + 1.00727647
    print('peptide_mz', peptide_mz)

    bare_sequence = whole[0]
    filtered_table, ion_mzs, ion_labels = filter_ion_table(tmp_ion_table,
                                                           bare_sequence,
                                                           sequence,
                                                           peptide_charge)

    fig, ax = plt.subplots()
    mzs = []
    exp_ints = []
    pred_ints = []
    labels = []
    plot_threshold = 0.05
    for j in range(0, ion_mzs.shape[0]):
        for k in range(0, ion_mzs.shape[1]):
            my_label = ion_labels[j, k]
            if exp_ions[j, k] > plot_threshold or pred_ions[j, k] > plot_threshold:
                if 'y' in my_label or 'b' in my_label:
                    mzs.append(ion_mzs[j, k])
                    exp_ints.append(exp_ions[j, k])
                    pred_ints.append(pred_ions[j, k])
                    labels.append(my_label)

    pred_ints = np.array(pred_ints) * -1
    ax.stem(mzs, exp_ints, 'b', markerfmt=" ", label='Experimental')
    ax.stem(mzs, pred_ints, 'g', markerfmt=" ", label='Predicted')
    for j in range(0, len(mzs)):
        if np.abs(pred_ints[j]) > exp_ints[j] and exp_ints[j] < 0.2:
            ax.annotate(labels[j], (mzs[j], pred_ints[j] - 0.03))
        else:
            ax.annotate(labels[j], (mzs[j], exp_ints[j]))

    plt.title(sequence + '/' + str(peptide_charge) )
    plt.text(250, -0.8, 'pcc: %6.3f, %6.3g'%(pcc[0], pcc[1]))
    plt.xlabel('m/z')
    plt.show()


fn = 'ucsd_hcd_splib_2ndhalf_ions.pickle'
with open(fn, 'rb') as fid:
    all_ions = pickle.load(fid)

n = 0
type_tag = 'test_ions'
plot_spectrum(all_ions, type_tag, n)
