# parameters
nearnum = 15

# get left and right sequences of the sites
def get_left_right_fre(proseq, t, cut_sites):
    global nearnum
    if len(proseq[:cut_sites[t]]) >= nearnum:
        leftfre = proseq[cut_sites[t]-nearnum:cut_sites[t]+1]
    else:
        leflen = len(proseq[:cut_sites[t]])
        leftfre = 'Z' * (nearnum - leflen) + proseq[:cut_sites[t]+1]
    if len(proseq[cut_sites[t]+1:]) >= nearnum:
        rightfre = proseq[cut_sites[t]+1:cut_sites[t]+1+nearnum]
    else:
        riglen = len(proseq[cut_sites[t]+1:])
        rightfre = proseq[cut_sites[t]+1:] + 'Z' * (nearnum-riglen)
    # leftfre='...K/R', rightfre='...', 
    # then use gener_train_seq to get left+right='...K/R...'
    return leftfre, rightfre

# get the full 31-mer sequences
def gener_train_seq(proseq, sites, ter, t, cut_sites):
    if t > 0 and t < len(cut_sites) - 1 and cut_sites[t] != cut_sites[0]:
        leftfre, righfre = get_left_right_fre(proseq, t, cut_sites)
        mer = leftfre + righfre
    else:
        mer = ''
    return mer

# get all theoretical peptides for each protein
def get_peptides(proseq, sites, ter, maxlen, minlen, cut_sites, n):
    # cut_sites is all the sites' location in the protein sequence, 
    # n is the maximum number of missed cleavge sites allowed in the peptides
    # get all possible 31-mers here, of all candidate sites, 
    # considering that the maximum number of  missed cleavage sites is n
    illegal = 'BJOUX'
    peptide = []
    length = len(cut_sites)
    for j in range(length - 1):
    # j --> location of the left site
        for i in range(1, length - j):
        # i --> location of the right site
            if i >= n + 2:
                break
            # peptides are different with different proteases
            if ter =='C':
                pep = proseq[cut_sites[j] + 1:cut_sites[j + i] + 1]
            else:
                pep = proseq[cut_sites[j]:cut_sites[j + i]]
            # if protein N-terminal is a cleavage site, pep will be '', 
            # and this loop should be skipped.
            lp = len(pep)
            if pep == '':
                break
            if lp >= minlen and lp <= maxlen:
                # 31-mer of left site
                leftclev = gener_train_seq(proseq, sites, ter, j, cut_sites)
                if len(set(illegal) - set(leftclev)) != 5:
                    continue
                # 31-mer of right site
                righclev = gener_train_seq(proseq, sites, ter, j+i, cut_sites)
                if len(set(illegal) - set(righclev)) != 5:
                    continue
                mediclev='||'
                # if i>1, then there is/are cleavage site/sites
                if i > 1:
                    # 31-mers of all the missed cleavage sites
                    for k in range(1, i):
                    # k is the number of missed cleavage sites
                        medileft, medirigh = get_left_right_fre(proseq, j + k, 
                                                                cut_sites)
                        submediclev = medileft + medirigh
                        mediclev = mediclev + submediclev + '||'
                if len(set(illegal) - set(mediclev)) != 5:
                    continue
                
                if leftclev != '':
                    pe = pep + '\t' + leftclev
                else:
                    pe = pep + '\t' + '*'

                if righclev != '':
                    pe += '\t' + righclev
                else:
                    pe += '\t' + '*'

                if mediclev != '':
                    pe += '\t' + mediclev
                else:
                    pe += '\t'
                peptide.append(pe)
                
                # N-terminal peptides starting with M have two conditions:
                # MPEPTIDESK | PEPTIDESK
                # the p(dig) of C-terminal site is 1 if cutting after 'M'
                if all([proseq[:lp] == pep, pep[0] == 'M', 
                        lp - 1 >= minlen, lp -1 <= maxlen]):
                    peptide.append(pe.lstrip('M'))
                
    return peptide

# protein --> peptide | left 31-mer | right 31-mer | missed 31-mers
def digestion(proseq, sites, ter, maxlen, minlen, missedcleavages):
    n = missedcleavages
    l = len(proseq)
    cut_sites = []
    peptides = []

    # mark the locations of the candidate cleavage sites
    for i in range(l):
        if proseq[i] in sites:
            cut_sites.append(i)
    
    # mark the foremost and end peptide
    # Attention!! there are some differences between C-terminal and N-terminal 
    # cleavage, but both cleavages need to start at the first amino acide and 
    # end at the last one.
    if ter == 'C':
        cut_sites.insert(0, -1)
        if cut_sites[-1] != l - 1:
            cut_sites.append(l - 1)
    elif ter == 'N':
        cut_sites.insert(0, 0)
        cut_sites.append(l)

    # see if there is any candidate sites to cut
    if len(cut_sites) > 2:
        peptides = get_peptides(proseq, sites, ter, maxlen, minlen, 
                                cut_sites, n)
    elif l <= maxlen and l >= minlen:
        print("Warning: Protein sequence %s has no candidate sites!"%proseq)
        peptides = [proseq]

    return peptides
