#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocessing.py: Processing of protein-peptide dataset to trainable dataset"""

__author__ = "Juho Son, Seungjin Na, Eunok Paek"
__copyright__ = "Copyright 2022, The DBDBDeep Project"
__credits__ = ["Juho Son", "Seungjin Na", "Eunok Paek"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Juho Son"
__email__ = "juho8563@hanyang.ac.kr"
__status__ = "Development"

import time


def get_spectral_cnt(data_path: str) -> dict():
    """
    Human Spectral Library(HSL) candidate to precursor count dictionary
    Input : Human Spectral Library (tsv file 9GB)
        - precursors : 30,633,841 (including all of HSL)
        - experiments : 27,992 (by filename. ex.mzXML)
    Output : precursor_cnt { (peptide, charge) : count } dictionary
    """
    precursor_cnt = dict()
    with open(data_path) as f:
        header = l = f.readline().replace('\n', '').split('\t')
        header_dic = {col:idx for idx, col in enumerate(header)}
        '''annotated_peak_count, annotation, augment_task, charge, explained_intensity,
        extract_task, filename, kl_score, most_similar_score, mz, number_of_ions_anotated_above_SNR,
        number_of_peaks_within_1_percent_of_max, ppm_error, precursor_intensity,
        proteosafe_task, scan, score'''
        
        while f:
            l = f.readline().replace('\n', '').split('\t')
            if l == ['']:
                break
            
            # get precursor(sequence, charge), filename
            seq = l[header_dic['annotation']]
            charge = l[header_dic['charge']]
            precursor = (seq, charge)
            filename = l[header_dic['filename']]
            if precursor not in precursor_cnt:  # init
                precursor_cnt[precursor] = set()
            precursor_cnt[precursor].add(filename)
        
        # precursor's spectral count by filename (experimental count)
        precursor_cnt = {k:len(v) for k, v in precursor_cnt.items()}
    return precursor_cnt


def get_precursor(data_path: str) -> dict():
    '''
    Human Spectral Library to identified proteins and precursors dictionary
    Input : Human Spectral Library (mgf file 8GB)
        - proteins : 19,611
        - precursors : 2,154,269 (PK consisting of SEQ and CHARGE)
        - strip peptides : 1,114,503
    Output : prot_precursor { protein : [ (peptide, charge)s ] } dict
    '''
    prot_precursor = dict()
    with open(data_path) as f:
        for line in f:
            l = line.replace('\n', '')
            # get peptide sequence, protein
            if 'CHARGE=' in l:
                pep_charge = l.replace('CHARGE=', '')
            elif 'SEQ=' in l:
                pep_seq = l.replace('SEQ=', '')
            elif 'PROTEIN=' in l:
                pep_prot_li = l.replace('PROTEIN=', '').split(';')
            # record precursor
            elif l == 'END IONS':
                precursor = (pep_seq, pep_charge)
                for pep_prot in pep_prot_li:
                    if 'XXX' not in pep_prot:  # exclude decoy protein
                        if pep_prot not in prot_precursor:
                            prot_precursor[pep_prot] = set()
                        prot_precursor[pep_prot].add(precursor)
        return prot_precursor


def merge_precursor_cnt(prot_precursor: dict, precursor_cnt: dict) -> dict():
    '''
    Merge Human Spectral Library(HSL) Candidate to HSL for spectral count per precursor
    Input : protein2precursor, precursor2count
    Output : protein2precursor_cnt { protein : [ (peptide, charge, spectral_cnt)s ] }
    '''
    prot_precursor_cnt = dict()
    for prot_name, precursors in prot_precursor.items():
        precursors_cnt = [(*precursor, precursor_cnt[precursor]) for precursor in precursors]
        prot_precursor_cnt[prot_name] = precursors_cnt
    return prot_precursor_cnt


def get_protein(data_path: str) -> dict():
    '''
    Human Uniprot to protein sequence dictionary
    Input : Human Uniprot (fasta file 30MB)
        - proteins : 75,074
    Output : { protein : sequence } dict
    '''
    prot_seq = dict()  # init
    seq = ''
    NAME_IDX = 0
    FIRST_LINE = True
    with open(data_path) as f:
        while f:
            l = f.readline().replace('\n', '')
            if l == '':
                prot_seq[prot_name] = seq
                break

            # get protein name and sequence
            if '>' in l:
                if FIRST_LINE:
                    prot_name = l.replace('>','').split(' ')[NAME_IDX]
                    FIRST_LINE = False
                    continue
                prot_seq[prot_name] = seq
                # new init for next protein
                prot_name = l.replace('>','').split(' ')[NAME_IDX]
                seq = ''
            else:
                seq += l
    return prot_seq


def merge_protein_precursor(prot_precursor_cnt: dict, prot_seq: dict) -> dict():
    '''
    Merge Uniprot and Human Spectral Library(HSL) for identified peptide counting
    Input : prot_precursor_cnt, prot_seq
    output : prot_seq_precursor { protein : {seq: ARNDCEQG, 
                                             precursor: [ (peptide, charge, spectral_cnt)s ]
                                            } 
                                }
    '''
    identified_prot = set(prot_precursor_cnt.keys())
    uniprot_prot = set(prot_seq.keys())
    inter_prot_names = identified_prot.intersection(uniprot_prot)
    prot_seq_precursor = dict()
    for prot_name in inter_prot_names:
        prot_seq_precursor[prot_name] = {'sequence':prot_seq[prot_name],
                                         'precursors':prot_precursor_cnt[prot_name]}
    return prot_seq_precursor


def get_peptide_cnt(prot_seq_precursor: dict) -> dict():
    '''
    Counting of striped peptide's spectral count (sum)
    Input : prot_seq_precursor
    Output : peptide_cnt { peptide : spectral_cnt } dictionary
    '''
    

def prot2cnt():
    pass

def pep_tree():
    pass

def prot_filtration(threshold=0.5):
    pass
# digest, detect protein


def pep_from_prot():
    pass


def pep2cnt():
    pass


def pep_detection_labelling():
    pass


def digest2detect():
    pass


def main():
    start_time = time.time()

    path = '/home/bis/2021_SJH_detectability/Detectability/data/'
    # [PEPTIDE] human spectral library candidate file
    hslcand_fn = 'massIVE-KB/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-candidate_library_spectra-main.tsv'
    # [PEPTIDE] human spectral library file
    hsl_fn = 'massIVE-KB/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf'
    # [PROTEIN] human uniprot file
    fasta_fn = 'uniprot/uniprot-proteome_UP000005640.fasta'

    print('### get spectral count ...\t\t\t')
    precursor_cnt = get_spectral_cnt(path + hslcand_fn)
    prot_precursor = get_precursor(path + hsl_fn)
    prot_precursor_cnt = merge_precursor_cnt(prot_precursor, precursor_cnt)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish getting of precursors ... time elapsed {elapsed_time} min \t\t\t')

    idx=0
    for k,v in prot_precursor:
        print(k,v)
        idx+=1
        if idx==5:
            break

    idx=0
    for k,v in prot_precursor_cnt:
        print(k,v)
        idx+=1
        if idx==5:
            break

    print('### get protein ... \t\t\t')
    prot_seq = get_protein(path + fasta_fn)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish getting of proteins ... time elapsed {elapsed_time} min \t\t\t')

    print('### merge files ... \t\t\t')
    prot_seq_precursor = merge_protein_precursor(prot_precursor_cnt, prot_seq)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish merging of proteins and precursors ... time elapsed {elapsed_time} min \t\t\t')

    print('intersection of massIVE-KB and Uniprot proteins : {}'.format(len(prot_seq_precursor)))

if __name__ == '__main__':
    main()
