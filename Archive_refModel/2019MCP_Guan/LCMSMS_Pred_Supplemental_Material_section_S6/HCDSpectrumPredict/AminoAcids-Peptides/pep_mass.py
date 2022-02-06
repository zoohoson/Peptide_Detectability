import sys
sys.path.insert(0, '..\\AminoAcids-Peptides')
from AAmass import *
from PTMmass import *
import re

def makeNtermLadder(sequence):
    n_ladder=[]
    pep_mass=std_aa_monoisotopic_mass['h']-std_aa_monoisotopic_mass['e']
    bare_seq,ptms=findAllPTM(sequence)
    if '-' in bare_seq:
        bare_seq=bare_seq.replace('-','')
    if -2 in ptms.keys():
        #has n-term mod
        pep_mass+=ptm_mono_mass[ptms[-2]]
        ladder_seq=ptms[-2]+'-'
    else:
        ladder_seq=''
    ptm_idx=1
    n_ladder.append([0,ladder_seq,pep_mass])
    for aa_idx,aa in enumerate(bare_seq):
        #if aa=='-':print sequence,bare_seq
        pep_mass+=std_aa_monoisotopic_mass[aa]
        ladder_seq+=aa
        if aa_idx in ptms.keys():
            pep_mass+=ptm_mono_mass[ptms[aa_idx]]
            ladder_seq+=ptms[aa_idx]
        n_ladder.append([aa_idx+1,ladder_seq,pep_mass])
    if -1 in ptms.keys():
        #has c-term mod
        pep_mass+=ptm_mono_mass[ptms[-1]]
        ladder_seq+='-'+ptms[-1]
    else:
        #terminate with OH
        pep_mass+=std_aa_monoisotopic_mass['o']+std_aa_monoisotopic_mass['h']
    whole=[bare_seq,ptms,pep_mass+std_aa_monoisotopic_mass['e']] 
    return whole,n_ladder

def makeCtermLadderBasedOnNLadder(sequence):
    whole,n_ladder=makeNtermLadder(sequence)
    
    monoMW=whole[-1]
    c_ladder=[]
    for p in range(1,len(n_ladder)):
        n_part=n_ladder[p]
        mass=monoMW-n_part[2]
        c_ladder.append([len(n_ladder)-n_part[0]-1,sequence.replace(n_part[1],''),mass])
        
    return whole,n_ladder,c_ladder

def parenthetic_contents(string):
    #http://stackoverflow.com/questions/4284991/parsing-nested-parentheses-in-python-grab-content-by-level
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), string[start: i+1])# changed from yield (len(stack), string[start + 1: i])
            #to include parentheses
            
def findAllPTM(sequence):
    #'(Acetyl)-SK(Phospho)EE(HexNaAc(2))AH(Phospho)AED(abc(xyz)123)SVM(Oxidation)DHHFR-(Methyl)'
    allLevels=list(parenthetic_contents(sequence))
    seq=sequence
    ptms=[]
    for item in allLevels:
        if item[0]==0:#get level 0 
            ptms.append([-1,item[1]])
    for j,ptm in enumerate(ptms):
        #print j, ptm, seq
        idx=seq.index(ptm[1])
        ptms[j][0]=idx
        seq=seq.replace(ptm[1],'',1)
    if seq[0]=='-':
        #has n-term mod
        ptms[0][0]=-2
        seq=seq[1:]
        hasPTMatNterm=True
    else:
        hasPTMatNterm=False
    if seq[-1]=='-':
        #has c-term mod
        ptms[-1][0]=-1
        seq=seq[:-1]
    for j,ptm in enumerate(ptms):
        if ptm[0]>0:
            ptms[j][0]-=1
            if hasPTMatNterm and ptms[j][0]>0:
                ptms[j][0]-=1
    my_ptms={}
    for item in ptms:
         my_ptms[item[0]]=item[1]
    return seq,my_ptms
    #my_ptms: key=position(starts at 1),vale=ptm
        
def cantHaveNeutralLoss(ion_type,seq_tag):
    h2o_loss=['S','R','E','D']
    nh3_loss=['R','H','K','N']
    if '-H2O' in ion_type:
        if not any(x in seq_tag for x in h2o_loss):
            return True
    if '-NH3' in ion_type:
        if not any(x in seq_tag for x in nh3_loss):
            return True
    return False

def getNeutralLoss(bare_seq,ptms):
    tmp=[]
    for idx in ptms:
        if idx==-2:
            #n-term
            tmp.append(ptms[idx]+'-')
        else:
            if idx==-1:
                #c-term
                tmp.append('-'+ptms[idx])
            else:
                tmp.append(bare_seq[idx]+ptms[idx])
    #print tmp
    tmp1=list(set(tmp))
    ptm_nls={}
    for ptm in tmp1:
        if ptm in ptm_neuralloss.keys():
            ptm_nls[getPTMfromStr(ptm)]=ptm_neuralloss[ptm]
    return ptm_nls    

def getPTMfromStr(ptm_str):
    l_dix=ptm_str.index('(')
    r_idx=ptm_str[::-1].index(')')
    return ptm_str[l_dix:len(ptm_str)-r_idx]

def makeParentIons(whole,parent_charge):
    mono_mass=whole[-1]
    ptms=whole[1]
    bare_seq=whole[0]
    
    ions=[]
    #proton_mass=std_aa_monoisotopic_mass['h']-std_aa_monoisotopic_mass['e']
    for charge in range(1,parent_charge+1):
        parent_mz=mono_mass/charge+proton_mass
        if charge>1:
            ions.append(['P^'+str(charge),parent_mz])
            ions.append(['P-H2O^'+str(charge),parent_mz+std_ion_type_monoiotopic_mass['y-H2O']/charge])
            ions.append(['P-NH3^'+str(charge),parent_mz+std_ion_type_monoiotopic_mass['y-NH3']/charge])
        else:
            ions.append(['P',parent_mz])
            ions.append(['P-H2O',parent_mz+std_ion_type_monoiotopic_mass['y-H2O']/charge])
            ions.append(['P-NH3',parent_mz+std_ion_type_monoiotopic_mass['y-NH3']/charge])

        ptm_nls=getNeutralLoss(bare_seq,ptms)
        for ptm in ptm_nls:
            if charge>1:
                ions.append(['P-'+ptm_nl_label[getPTMfromStr(ptm)]+'^'+str(charge),parent_mz-ptm_nls[ptm]/charge])
            else:
                ions.append(['P-'+ptm_nl_label[getPTMfromStr(ptm)],parent_mz-ptm_nls[ptm]/charge])
        ptm_nls['']=0.#add a dummy neutral loss
    return ions,ptm_nls
   
def makeSequenceIons(ion_types,sequence,parent_charge,mz_range):   
    whole,n_ladder,c_ladder=makeCtermLadderBasedOnNLadder(sequence)
    ions,ptm_nls=makeParentIons(whole,parent_charge)
    charges=range(1,parent_charge+1)
    for ion_type in ion_types:
        if ion_type[0] in ['a','b','c']:
            my_ladder=n_ladder[1:]#n-term ion
        else:
            #['x','y','z']:#c-term ion
            my_ladder=c_ladder
        for my_la in my_ladder:
            for charge in charges:          
                nl_label=''
                nl_mass=0.
                for ptm in ptm_nls:
                    if len(ptm)>0:
                        if ptm in my_la[1]:
                            nl_label='-'+ptm_nl_label[ptm]
                    nl_mass=ptm_nls[ptm]
                    ion=ion_type[0]+str(my_la[0])+ion_type[1:]+nl_label
                    if charge>1:
                        ion+='^'+str(charge)
                    mass=my_la[-1]+std_ion_type_monoiotopic_mass[ion_type]+proton_mass*(charge-1.0)-nl_mass
                    mz=mass/charge
                    if mz>mz_range[0] and mz<mz_range[1]:
                        #if not cantHaveNeutralLoss(ion_type,my_la[1]):
                        ions.append([ion,mz,my_la[1]])
    return whole,ions

def calculateMonoisotopicMW(sequence):
    n_ladder=makeNtermLadder(sequence)
    return n_ladder[-1][-1]+std_aa_monoisotopic_mass['o']+std_aa_monoisotopic_mass['h']

def removeDuplicates(my_list):
    no_duplicates=[]
    while len(my_list)>0:
        tmp=my_list.pop()
        if tmp not in my_list:
            no_duplicates.append(tmp)
    return no_duplicates    

def get_all_internal_substrings(input_string):
  length = len(input_string)
  return [input_string[i:j+1] for i in xrange(1,length-1) for j in xrange(i,length-1)]

def makeInterIons(sequence):
    whole,n_ladder,c_ladder=makeCtermLadderBasedOnNLadder(sequence)
    mono_mass=whole[-1]
    ptms=whole[1]
    bare_seq=whole[0]
    
    Internal_ions=[]
    for jn,sn,mn in n_ladder:
        if jn>=1 and jn<len(bare_seq)-1:
            for jc,sc,mc in c_ladder:
                if jc>=1 and jc<len(bare_seq)-1 and jn+jc+1<len(bare_seq):
                    Internal_ions.append([sequence[len(sn):len(sequence)-len(sc)],\
                                          mono_mass-mn-mc+std_aa_monoisotopic_mass['h']-\
                                          std_aa_monoisotopic_mass['e']])
    return removeDuplicates(Internal_ions)

def makeImmoniumIons(whole):
    bare_seq=whole[0]
    ImIons=[]
    for aa in std_immonium_ions.keys():
        if aa in bare_seq:
            ImIons.append([aa,std_immonium_ions[aa]])
    return ImIons

def sequence2composition(sequence):
    #terminal formula
    composition = {'H':2,'O':1}
    
    for aa in sequence:
        aa_comp = std_aa_comp[aa]
        for element in aa_comp:
            if element not in composition:
                composition[element] = 0
            composition[element] = composition[element] + aa_comp[element]

    return composition

def msgfSeq2composition(msgf_sequence, ptm_compositions):
    modXsequence = convertPTMinSequence2modX(msgf_sequence)
    bare_sequence, ptms = findAllPTM(modXsequence)
    composition = sequence2composition(bare_sequence)
    for ptm_key in ptms:
        ptm = ptms[ptm_key]
        ptm_comp = ptm_compositions[ptm]
        for element in ptm_comp:
            composition[element] = composition[element] + ptm_comp[element]
            
    return bare_sequence, ptms, composition
    
##HCD_ion_types=['a','a-H2O','a-NH3','b','b-H2O','b-NH3','y','y-H2O','y-NH3']
##HCD_ion_types=['b','y']
##sequence='TPSLPT(Phospho)PPTR'
##sequence='(Gln->pyro-Glu)-QDPPSVVVTSHQAPGEK'
##sequence='(Acetyl)-SDKPDM(Oxidation)AEIEKFDK'
##parent_charge=3
##whole,ions=makeSequenceIons(HCD_ion_types,sequence,parent_charge,(100.,2000.))

