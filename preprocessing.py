# -*- coding: utf-8 -*-

"""preprocessing.py: Processing of protein-peptide dataset to trainable dataset"""

__author__ = "Juho Son, Seungjin Na, Eunok Paek"
__copyright__ = "Copyright 2022, The DbyDeep Project"
__credits__ = ["Juho Son", "Seungjin Na", "Eunok Paek"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Juho Son"
__email__ = "juho8563@hanyang.ac.kr"
__status__ = "Development"


import time
import pandas as pd
import json
from ahocorapy.keywordtree import KeywordTree


class preprocessing():
    def __init__(self):
        self.data = ''



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
    output : prot_seq_precursor { protein : {sequence: ARNDCEQG, 
                                             precursors: [ (peptide, charge, spectral_cnt)s ]
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
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           peptides: { peptide : count }
                                          } 
                                }
    '''
    prot_seq_peptide = dict()
    for prot_name, prot_info in prot_seq_precursor.items():
        prot_seq = prot_info['sequence']
        prot_precursors = prot_info['precursors']

        prot_pep_cnt = dict()
        for precursor in prot_precursors:
            pre_ptm_seq, pre_charge, pre_cnt = precursor
            pre_strip_seq = ''.join(filter(lambda x: ord(x) in range(65, 91), pre_ptm_seq))
            if pre_strip_seq not in prot_pep_cnt:
                prot_pep_cnt[pre_strip_seq] = 0
            prot_pep_cnt[pre_strip_seq] += pre_cnt
        if prot_name not in prot_seq_peptide:
            prot_seq_peptide[prot_name] = dict()
        prot_seq_peptide[prot_name]['sequence'] = prot_seq
        prot_seq_peptide[prot_name]['peptides'] = prot_pep_cnt
    return prot_seq_peptide


def get_peptide_tree(prot_seq_peptide: dict) -> dict():
    '''
    Contructing peptide tree
    Input : prot_seq_peptide
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           peptides: { peptide : count }
                                           peptide_tree: tree_instance
                                          } 
                                }
    '''
    for prot_name, prot_info in prot_seq_peptide.items():
        prot_peptideas = prot_info['peptides']
        tree = KeywordTree()
        for peptide in prot_peptideas:
            tree.add(peptide)
        tree.finalize()
        prot_seq_peptide[prot_name]['peptide_tree'] = tree
    return prot_seq_peptide


def get_protein_coverage(prot_seq_peptide: dict) -> dict():
    '''
    Contructing peptide tree
    Input : prot_seq_peptide
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           peptides: { peptide : count }
                                           peptide_tree: tree_instance,
                                           coverage: 0.5
                                          } 
                                }
    '''
    for prot_name, prot_info in prot_seq_peptide.items():
        prot_seq = prot_info['sequence']
        tree = prot_info['peptide_tree']

        prot_coverage = [0 for _ in prot_seq]
        prot_peptides = tree.search_all(prot_seq)
        for peptide, START_IDX in prot_peptides:  # result = ('ACAC', 5) = (peptide, location) by ahocorapy
            for AA_IDX, cnt in enumerate(prot_coverage):
                if AA_IDX in range(START_IDX, START_IDX + len(peptide)):
                    cnt += 1
                    prot_coverage[AA_IDX] = cnt

        IDENTIFIED_AMINO_ACID_NUM = sum([1 for AA_CNT in prot_coverage if AA_CNT >= 1])
        PROTEIN_LENGTH = len(prot_seq)
        COVERAGE = IDENTIFIED_AMINO_ACID_NUM / PROTEIN_LENGTH
        prot_seq_peptide[prot_name]['coverage'] = COVERAGE
    return prot_seq_peptide


def get_protein_count(prot_seq_peptide: dict) -> dict():
    '''
    Counting of protein's amino acid with identified peptide's spectral count
    Input : prot_seq_peptide
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           cleavage_cnt: [1, 0, ..., 4],
                                           miss_cleavage_cnt: [0, 1, ..., 0],
                                           peptides: { peptide : count }
                                           peptide_tree: tree_instance,
                                           coverage: 0.5
                                           }
                                }
                                         
    '''
    TRYPTIC_SITE = 'KR'
    for prot_name, prot_info in prot_seq_peptide.items():
        prot_seq = prot_info['sequence']
        pep_tree = prot_info['peptide_tree']
        pep_cnt = prot_info['peptides']
        # counting array init
        prot_cleavage_cnt = [0 for _ in prot_seq]
        prot_miss_cleavage_cnt = [0 for _ in prot_seq]
        
        prot_peptides = pep_tree.search_all(prot_seq)
        for peptide, START_IDX in prot_peptides: 
            SPECTRAL_CNT = pep_cnt[peptide]
            CLEAVAGE_CNT = sum([1 for amino_acid in peptide[:-1] if amino_acid in TRYPTIC_SITE])  # except C-terminal of peptide
            # Case of inlcuding missed cleavage sites (__KR__)
            if CLEAVAGE_CNT >= 1:  
                
                # Case of that C-terminal of peptide is cleavage site  (__KR__KR)
                if peptide[-1] in TRYPTIC_SITE:
                    # Counting of missed cleavage sites at amino acid level
                    for AA_IDX, value in enumerate(prot_miss_cleavage_cnt):
                        if AA_IDX in range(START_IDX, START_IDX + len(peptide) -1):
                            value += SPECTRAL_CNT
                            prot_miss_cleavage_cnt[AA_IDX] = value
                    # Counting of cleavage sites at amino acid level
                    # only count first and last amino acid of peptide, which include miss cleavage
                    # last : For counting N-terminal of cleavage site
                    # first : For counting C-terminal of cleavage site
                    prot_cleavage_cnt[START_IDX + len(peptide) - 1] += SPECTRAL_CNT
                    prot_cleavage_cnt[START_IDX] += SPECTRAL_CNT
                
                # Case of that C-terminal of peptide is not cleavage site (__KR__)
                else:
                    # Counting of missed cleavage sites at amino aicd level
                    for AA_IDX, value in enumerate(prot_miss_cleavage_cnt):
                        if AA_IDX in range(START_IDX, START_IDX + len(peptide)):
                            value += SPECTRAL_CNT
                            prot_miss_cleavage_cnt[AA_IDX] = value
            
            # Case of not including missed cleavage sites
            else:
                # peptide(spectral) count
                for AA_IDX, value in enumerate(prot_cleavage_cnt):
                    if AA_IDX in range(START_IDX, START_IDX + len(peptide)):
                        value += pep_cnt[peptide]
                        prot_cleavage_cnt[AA_IDX] = value

        prot_seq_peptide[prot_name]['cleavage_cnt'] = prot_cleavage_cnt
        prot_seq_peptide[prot_name]['miss_cleavage_cnt'] = prot_miss_cleavage_cnt
    
    return prot_seq_peptide


def filter_protein(prot_seq_peptide: dict, threshold: float = 0.5) -> dict():
    '''
    Filtering of protein which has coverage above 0.5
    Input : prot_seq_peptide
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           cleavage_cnt: [1, 0, ..., 4],
                                           miss_cleavage_cnt: [0, 1, ..., 0],
                                           peptides: { peptide : count }
                                           peptide_tree: tree_instance,
                                           coverage: 0.5
                                           }
                                }

    '''
    filtered_prot_seq_peptide = dict()
    all_peptides = set()
    filtered_peptides = set()
    for prot_name, prot_info in prot_seq_peptide.items():
        ##################################
        # counting for paper
        for peptide in prot_info['peptides']:
            all_peptides.add(peptide)
        ##################################
        if prot_info['coverage'] >= 0.5:
            filtered_prot_seq_peptide[prot_name] = prot_info
            ##################################
            for peptide in prot_info['peptides']:
                filtered_peptides.add(peptide)
            ##################################
    print('##################################')
    print('whole numboer of identified proteins, peptides :', 
            len(prot_seq_peptide), len(all_peptides))
    print('whole numboer of filtered proteins, peptides :', 
            len(filtered_prot_seq_peptide), len(filtered_peptides))
    print('##################################')
    return filtered_prot_seq_peptide


def get_peptide_from_protein(prot_seq_peptide: dict, DIGEST_MERS:int = 7) -> dict():
    '''
    Getting peptides from proteins
    Input : prot_seq_peptide
    Output : prot_seq_peptide { protein : {sequence: ARNDCEQG, 
                                           cleavage_cnt: [1, 0, ..., 4],
                                           miss_cleavage_cnt: [0, 1, ..., 0],
                                           peptides: { peptide : count }
                                           peptide_tree: tree_instance,
                                           coverage: 0.5,
                                           labelled_peptides : { peptide : {n: AA,
                                                                            c: AA,
                                                                            m1: AA,
                                                                            m2: AA
                                                                }
                                           }
                                }
    '''
    
    TRYPTIC_SITE = 'KR'
    for prot_name, prot_info in prot_seq_peptide.items():
        # for labelling
        pep_tree = prot_info['peptide_tree']
        pep_cnt = prot_info['peptides']

        # slicing
        prot_seq = prot_info['sequence']
        prot_seq_peptide[prot_name]['labelled_peptides'] = dict()  # init
        cleavage_sites = [AA_IDX for AA_IDX, aa in enumerate(prot_seq) if aa in TRYPTIC_SITE]
        MISS_CLEAVAGES = [0, 1, 2]
        for MISS_CLEAVAGE_CNT in MISS_CLEAVAGES:

            # consider miss cleavage sites and c-terminal site
            cleavage_range = range(len(cleavage_sites) - MISS_CLEAVAGE_CNT - 1)  
            for CS_IDX in cleavage_range:
                

                # N terminal of peptide
                condi_prot_nterm_pep_nterm = cleavage_sites[CS_IDX] < DIGEST_MERS
                condi_prot_cterm_pep_nterm = cleavage_sites[CS_IDX] > len(prot_seq) - 1 - (DIGEST_MERS)
                nterm_pad_pep_nterm = 'Z' * (DIGEST_MERS - cleavage_sites[CS_IDX]) 
                cterm_pad_pep_nterm = 'Z' * (DIGEST_MERS - (len(prot_seq) - 1 - cleavage_sites[CS_IDX]))
                pep_nterm_start_idx = cleavage_sites[CS_IDX] - DIGEST_MERS
                pep_nterm_end_idx = cleavage_sites[CS_IDX] + DIGEST_MERS + 1
                if condi_prot_nterm_pep_nterm and condi_prot_cterm_pep_nterm:  # ex) --MNQKLLK-- : both n and c term of protein are insufficient
                    # pep_seq = prot_seq[:pep_nterm_end_idx]  # test ...
                    pep_seq = prot_seq
                    pep_nterm = nterm_pad_pep_nterm + pep_seq + cterm_pad_pep_nterm
                elif condi_prot_nterm_pep_nterm:
                    pep_seq = prot_seq[:pep_nterm_end_idx]
                    pep_nterm = nterm_pad_pep_nterm + pep_seq
                elif condi_prot_cterm_pep_nterm:
                    pep_seq = prot_seq[pep_nterm_start_idx:]
                    pep_nterm = pep_seq + cterm_pad_pep_nterm
                else:
                    pep_nterm = prot_seq[pep_nterm_start_idx : pep_nterm_end_idx]
                
                # C terminal of peptide
                condi_prot_nterm_pep_cterm = cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1] < DIGEST_MERS
                condi_prot_cterm_pep_cterm = cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1] > len(prot_seq) - 1 - (DIGEST_MERS)
                nterm_pad_pep_cterm = 'Z' * (DIGEST_MERS - cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1])
                cterm_pad_pep_cterm = 'Z' * (DIGEST_MERS - (len(prot_seq) - 1 - cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1]))
                pep_cterm_start_idx = cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1] - DIGEST_MERS
                pep_cterm_end_idx = cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1] + (DIGEST_MERS + 1)
                if condi_prot_nterm_pep_cterm and condi_prot_cterm_pep_cterm:
                    # pep_seq = prot_seq[:pep_cterm_end_idx]  # test ...
                    pep_seq = prot_seq
                    pep_cterm = nterm_pad_pep_cterm + pep_seq + cterm_pad_pep_cterm
                elif condi_prot_nterm_pep_cterm:
                    pep_seq = prot_seq[:pep_cterm_end_idx]
                    pep_cterm = nterm_pad_pep_cterm + pep_seq
                elif condi_prot_cterm_pep_cterm:
                    pep_seq = prot_seq[pep_cterm_start_idx:]
                    pep_cterm = pep_seq + cterm_pad_pep_cterm
                else:
                    pep_cterm = prot_seq[pep_cterm_start_idx : pep_cterm_end_idx]

                # missed cleavage site of peptide
                pep_miss = {'pep_miss1': 'Z' * DIGEST_MERS + 'Z' + 'Z' * DIGEST_MERS,  # init
                            'pep_miss2': 'Z' * DIGEST_MERS + 'Z' + 'Z' * DIGEST_MERS}
                for mcc in range(1, MISS_CLEAVAGE_CNT + 1):
                    condi_prot_nterm_pep_miss = cleavage_sites[CS_IDX + mcc] < DIGEST_MERS
                    condi_prot_cterm_pep_miss = cleavage_sites[CS_IDX + mcc] > len(prot_seq) - 1 - (DIGEST_MERS)
                    nterm_pad_pep_miss = 'Z' * (DIGEST_MERS - cleavage_sites[CS_IDX + mcc])
                    cterm_pad_pep_miss = 'Z' * (DIGEST_MERS - (len(prot_seq) - 1 - cleavage_sites[CS_IDX + mcc]))
                    pep_miss_start_idx = cleavage_sites[CS_IDX + mcc] - DIGEST_MERS
                    pep_miss_end_idx = cleavage_sites[CS_IDX + mcc] + DIGEST_MERS + 1
                    if condi_prot_nterm_pep_miss and condi_prot_cterm_pep_miss:
                        pep_seq = prot_seq
                        pep_miss['pep_miss' + str(mcc)] = nterm_pad_pep_miss + pep_seq + cterm_pad_pep_miss
                    if condi_prot_nterm_pep_miss:
                        pep_seq = prot_seq[:pep_miss_end_idx]
                        pep_miss['pep_miss' + str(mcc)] = nterm_pad_pep_miss + pep_seq
                    elif condi_prot_cterm_pep_miss:
                        pep_seq = prot_seq[pep_miss_start_idx:]
                        pep_miss['pep_miss' + str(mcc)] = pep_seq + cterm_pad_pep_miss
                    else:
                        pep_miss['pep_miss' + str(mcc)] = prot_seq[pep_miss_start_idx : pep_miss_end_idx]
                pep_miss1, pep_miss2 = pep_miss['pep_miss1'], pep_miss['pep_miss2']

                # peptide
                pep_body_start_idx = cleavage_sites[CS_IDX] + 1
                pep_body_end_idx = cleavage_sites[CS_IDX + MISS_CLEAVAGE_CNT + 1] + 1
                pep_body = prot_seq[pep_body_start_idx : pep_body_end_idx]
                if not 6 <= len(pep_body) <= 40:  # following identified peptide (massIVE-KB) length distribution
                    continue  # not save this peptide

                # filtration of U, X (Amino Acid)
                ux_filter = lambda seq: True if 'U' in seq or 'X' in seq else False
                if ux_filter(pep_body) or ux_filter(pep_nterm) or ux_filter(pep_cterm) or ux_filter(pep_miss1) or ux_filter(pep_miss2):
                    continue  # not save this peptide

                # labelling
                PEP_IDX = 0
                PEP_START_IDX = 1
                identified_peps = [p[PEP_IDX] for p in pep_tree.search_all(pep_body)]
                if pep_body in identified_peps:
                    if pep_cnt[pep_body] > 1:  # positive : peptide has spectral count over 1 (at least 2)
                        label = True
                    else:
                        continue  # not save this peptide (neither positive nor negative peptide)
                else:  # negative : not identified peptide
                    label = False
                
                # save
                pep_set = ' '.join([pep_body, pep_nterm, pep_cterm, pep_miss1, pep_miss2])
                prot_seq_peptide[prot_name]['labelled_peptides'][pep_set] = label

    return prot_seq_peptide


def divide_digestability_detectability(prot_seq_peptide: dict) -> dict():
    '''
    dividing peptides for compare with AP3
    Input : prot_seq_peptide
    Output : prot_digest, prot_detect (same with input, just divide)
    '''
    
    prot_name_idx = {pn:idx for idx, pn in enumerate(prot_seq_peptide.keys())}
    prot_name_idx_rev = {idx:pn for pn, idx in prot_name_idx.items()}
    digest_prot_name = {prot_name_idx_rev[digest_idx] for digest_idx in range(len(prot_name_idx_rev)//2)}
    detect_prot_name = {prot_name_idx_rev[detect_idx] for detect_idx in range(len(prot_name_idx_rev)//2, len(prot_name_idx_rev))}
    prot_digest = {pn:dict(prot_seq_peptide[pn]) for pn in digest_prot_name}
    prot_detect = {pn:dict(prot_seq_peptide[pn]) for pn in detect_prot_name}
    return prot_digest, prot_detect
    
    
def merge_digestability_detectability(prot_digest: dict, prot_detect: dict) -> dict():
    '''
    Merging peptide to dataframe
    Input : prot_seq_peptide
    Output : pandas dataframe (peptide, nterm, cterm, miss1, miss2, label columns)
    '''

    for_detect_peptide_dupli = dict()
    for prot_name, prot_info in prot_detect.items():
        for pep_set, label in prot_info['labelled_peptides'].items():
            p, n, c, m1, m2 = pep_set.split()
            pep_set = (p, n, c, m1, m2)
            if pep_set not in for_detect_peptide_dupli:
                for_detect_peptide_dupli[pep_set] = set()
            for_detect_peptide_dupli[pep_set].add(label)
    for_detect_peptide = set()
    for pep_set, labels in for_detect_peptide_dupli.items():
        label = max(labels)
        p, n, c, m1, m2 = pep_set
        for_detect_peptide.add((p, n, c, m1, m2, label))

    for_digest_peptide_dupli = dict()
    for prot_name, prot_info in prot_digest.items():
        for pep_set, label in prot_info['labelled_peptides'].items():
            p, n, c, m1, m2 = pep_set.split()
            pep_set = (p, n, c, m1, m2)
            if pep_set not in for_digest_peptide_dupli:
                for_digest_peptide_dupli[pep_set] = set()
            for_digest_peptide_dupli[pep_set].add(label)
    for_digest_peptide = set()
    for pep_set, labels in for_digest_peptide_dupli.items():
        label = max(labels)
        p, n, c, m1, m2 = pep_set
        for_digest_peptide.add((p, n, c, m1, m2, label))

    df_detect = pd.DataFrame(for_detect_peptide,
                             columns = ['peptide', 'nterm', 'cterm', 'miss1', 'miss2', 'label'])
    df_digest = pd.DataFrame(for_digest_peptide,
                             columns = ['peptide', 'nterm', 'cterm', 'miss1', 'miss2', 'label'])
    
    # remove duplicated peptide context
    df_detect.drop_duplicates(inplace=True, ignore_index=True)
    df_digest.drop_duplicates(inplace=True, ignore_index=True)
    df_data = pd.concat([df_detect, df_digest], axis=0).reset_index(drop=True)
    df_data_key = df_data.drop(['label'], axis=1)
    detect_idx = set(df_data.iloc[:len(df_detect)].index)
    digest_idx = set(df_data.iloc[len(df_detect):].index)
    df_data_key.drop_duplicates(inplace=True, ignore_index=False)
    remain_idx = set(df_data_key.index)
    detect_idx = remain_idx.intersection(detect_idx)
    digest_idx = remain_idx.intersection(digest_idx)
    df_detect = df_data.loc[detect_idx]
    df_digest = df_data.loc[digest_idx]

    # divide train test dataset
    df_test = df_detect.sample(frac = 0.4, random_state = 2022)
    df_train_detect = df_detect.drop(df_test.index, axis=0)
    df_train = pd.concat([df_train_detect, df_digest], axis=0).reset_index(drop=True)

    # remove shared peptide in train dataset
    df_key = pd.DataFrame([[p, True] for p in df_test['peptide'].unique()],
                          columns = ['peptide', 'drop'])
    df_train = df_train.merge(df_key, on='peptide', how='left')
    df_train = df_train.loc[df_train['drop'].isnull()].drop('drop', axis=1).reset_index(drop=True)

    # sampling of negative data
    positive_tmp = df_test.loc[df_test.label==True]
    negative_tmp = df_test.loc[df_test.label==False].sample(len(positive_tmp), random_state=2022)
    df_test_sampled = pd.concat([positive_tmp, negative_tmp] ,axis=0).reset_index(drop=True)

    positive_tmp = df_train.loc[df_train.label==True]
    negative_tmp = df_train.loc[df_train.label==False].sample(len(positive_tmp), random_state=2022)
    df_train_sampled = pd.concat([positive_tmp, negative_tmp] ,axis=0).reset_index(drop=True)

    return df_train_sampled, df_test_sampled, df_train, df_test


def main():
    start_time = time.time()

    path = '/home/bis/2021_SJH_detectability/Detectability/data/'
    # [PEPTIDE] human spectral library candidate file
    hslcand_fn = 'massIVE-KB/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-candidate_library_spectra-main.tsv'
    # [PEPTIDE] human spectral library file
    hsl_fn = 'massIVE-KB/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf'
    # [PROTEIN] human uniprot file
    fasta_fn = 'uniprot/uniprot-proteome_UP000005640.fasta'

    # preprocessing of massIVE-KB
    print('### get spectral count ...\t\t\t', end='\r')
    precursor_cnt = get_spectral_cnt(path + hslcand_fn)
    prot_precursor = get_precursor(path + hsl_fn)
    prot_precursor_cnt = merge_precursor_cnt(prot_precursor, precursor_cnt)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish getting of precursors ... time elapsed {elapsed_time} min \t\t\t')

    # preprocessing of Uniprot
    print('### get protein ... \t\t\t', end='\r')
    prot_seq = get_protein(path + fasta_fn)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish getting of proteins ... time elapsed {elapsed_time} min \t\t\t')

    # preprocessing of massIVE-KB and Uniprot
    print('### merge files ... \t\t\t', end='\r')
    prot_seq_precursor = merge_protein_precursor(prot_precursor_cnt, prot_seq)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish merging of proteins and precursors ... time elapsed {elapsed_time} min \t\t\t')

    # preprocessing of dataset for train
    print('### matching proteins and peptides ... \t\t\t', end='\r')
    prot_seq_peptide = get_peptide_cnt(prot_seq_precursor)
    prot_seq_peptide = get_peptide_tree(prot_seq_peptide)
    prot_seq_peptide = get_protein_coverage(prot_seq_peptide)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish matching proteins and peptides ... time elapsed {elapsed_time} min \t\t\t')

    # Counting spectral count
    print('### Counting of peptide\'s spectral experiments ... \t\t\t', end='\r')
    prot_seq_peptide = get_protein_count(prot_seq_peptide)
    prot_seq_peptide = filter_protein(prot_seq_peptide)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish counting and Labelling ... time elapsed {elapsed_time} min \t\t\t')

    # Getting peptides from protein
    print('### Getting peptides from proteins ... \t\t\t', end='\r')
    prot_seq_peptide = get_peptide_from_protein(prot_seq_peptide)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish getting and labelling of peptides ... time elapsed {elapsed_time} min \t\t\t')

    print('### Dividing and Merging for train, val, test ... \t\t\t', end='\r')
    prot_digest, prot_detect = divide_digestability_detectability(prot_seq_peptide)
    df_train, df_test, df_train_whole, df_test_whole = merge_digestability_detectability(prot_digest, prot_detect)
    df_train.to_csv('data/train.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)
    elapsed_time = int((time.time()-start_time)/60)
    print(f'### finish merging peptide to train, val, and test dataset ... time elapsed {elapsed_time} min \t\t\t')
    
    data_tree = open('data/data.json', 'w')
    new = dict()
    for pn, pi in prot_seq_peptide.items():
        del pi['peptide_tree']
        new[pn] = pi
    json.dump(new, data_tree)
    data_tree.close()

    tmp = open('data/data_digestibility_AP3.json', 'w')
    new = dict()
    for pn, pi in prot_digest.items():
        del pi['peptide_tree']
        new[pn] = pi
    json.dump(new, tmp)
    tmp.close()

    tmp = open('data/data_detectability_AP3.json', 'w')
    new = dict()
    for pn, pi in prot_detect.items():
        del pi['peptide_tree']
        new[pn] = pi
    json.dump(new, tmp)
    tmp.close()

if __name__ == '__main__':
    main()
