##Prediction of LCMSMS properties of peptides from sequence by deep learning
##Shenheng Guan1, Michael F. Moran, and Bin Ma
##2019-02-21

import numpy as np

psi_to_single_ptm = {'(Acetyl)-': 'B',
                     '(Carbamyl)': 'O',
                     '(Carbamidomethyl)': '',
                     'M(Oxidation)': 'J',
                     '(Gln->pyro-Glu)Q': 'X',
                     'N(Deamidated)': 'D',
                     'Q(Deamidated)': 'E'}

def reshapeOneHot(X):
    X = np.dstack(X)
    X = np.swapaxes(X, 1, 2)
    X = np.swapaxes(X, 0, 1)
    return X

def get_single_ptm_code(psi_sequence):
    sequence = psi_sequence
    for ptm in psi_to_single_ptm:
        sequence = sequence.replace(ptm, psi_to_single_ptm[ptm])
    return sequence

def one_hot_encode_peptide(psi_sequence, MAX_LENGTH = 40):
    peptide = get_single_ptm_code(psi_sequence)
    if len(peptide) > MAX_LENGTH:
        print('Peptide length is larger than maximal length of ', str(MAX_LENGTH))
        return None
    else:
        AA_vocabulary = 'KRPTNAQVSGILCMJHFYWEDBXOU'#B: acetyl; O: Carbamyl; J: oxidized Met; X:pyro_glu
        no_not_used_aas = 1#U: not used

        one_hot_peptide = np.zeros((len(peptide), len(AA_vocabulary) - no_not_used_aas))

        for j in range(0, len(peptide)):
            try:
                aa = peptide[j]
                one_hot_peptide[j, AA_vocabulary.index(aa)] = 1
            except:
                pass
        
        no_front_paddings = int((MAX_LENGTH - len(peptide))/2)
        peptide_front_paddings = np.zeros((no_front_paddings, one_hot_peptide.shape[1]))

        no_back_paddings = MAX_LENGTH - len(peptide) - no_front_paddings
        peptide_back_paddings = np.zeros((no_back_paddings, one_hot_peptide.shape[1]))

        full_one_hot_peptide = np.vstack((peptide_front_paddings, one_hot_peptide, peptide_back_paddings))

        return peptide, full_one_hot_peptide
    
def one_hot_encode_peptide_ion(psi_sequence, charge, MAX_LENGTH = 40, MAX_CHARGE = 6):

    peptide, full_one_hot_peptide = one_hot_encode_peptide(psi_sequence)
    
    one_hot_charge = np.zeros((len(peptide), MAX_CHARGE))
    one_hot_charge[:, charge - 1] = 1
    
    no_front_paddings = int((MAX_LENGTH - len(peptide))/2)
    charge_front_paddings = np.zeros((no_front_paddings, one_hot_charge.shape[1]))

    no_back_paddings = MAX_LENGTH - len(peptide) - no_front_paddings
    charge_back_paddings = np.zeros((no_back_paddings, one_hot_charge.shape[1]))

    full_one_hot_charge = np.vstack((charge_front_paddings, one_hot_charge, charge_back_paddings))

    full_one_hot_peptide_ion = np.hstack((full_one_hot_peptide, full_one_hot_charge))

    return full_one_hot_peptide_ion

seq = '(Acetyl)-AC(Carbamidomethyl)DEM(Oxidation)NN(Deamidated)QQ(Deamidated)K'
ohc_seq = one_hot_encode_peptide(seq)
ohc_ion = one_hot_encode_peptide_ion(seq, 2)
