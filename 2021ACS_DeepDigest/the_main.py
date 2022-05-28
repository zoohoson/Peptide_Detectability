import time
start = time.time()
import sys
import getopt
import read_protein_sequences
import cut_and_gene_tradata
import predictor


def info(protease):
    # protease --> cleavage sites and terminal
    if protease == 'Trypsin':
        sites = ['K', 'R']
        ter = 'C'
    elif protease == 'LysargiNase':
        sites = ['K', 'R']
        ter = 'N'
    elif protease == 'LysC':
        sites = ['K']
        ter = 'C'
    elif protease == 'LysN':
        sites = ['K']
        ter = 'N'
    elif protease == 'ArgC':
        sites = ['R']
        ter = 'C'
    elif protease == 'Chymotrypsin':
        sites = ['W', 'F', 'Y', 'L', 'M']
        ter = 'C'
    elif protease == 'GluC':
        sites = ['E'] # sites = ['E', 'D']
        ter = 'C'
    elif protease == 'AspN':
        sites = ['D']
        ter = 'N'
    else:
        print("Error: This tool does not support %s protease yet,"
              " only Trypsin, ArgC, Chymotrypsin, GluC, LysC, AspN,"
              " LysN and LysargiNase are optional for now."%protease)
        sys.exit(1)
    return sites, ter

def AI_tool(datapath, resultpath, regular, protease, 
            maxlen, minlen, missedcleavages):
    sites, ter = info(protease)
    s1 = time.time()
    print("Reading file...")
    s = read_protein_sequences.read_protein_sequences(datapath, regular)
    e1 = time.time()
    print("Time cost of loading file was %s seconds."%(e1-s1))

    s2 =time.time()
    print("In silico digesting...")
    data =[]
    # get possible digestied peptides for each protein
    for protein in s:
        proid = protein[0]
        proseq = protein[1]
        peptide_list = cut_and_gene_tradata.digestion(proseq, sites, ter, 
                                                      maxlen, minlen, 
                                                      missedcleavages)
        for peptide in peptide_list:
            data.append([proid, peptide])
    # np.savetxt("peptides.txt", data, fmt='%s\t%s', 
                # delimiter='\t', newline='\n', comments='')
    e2 = time.time()
    print("Time cost of in silico digestion was %s seconds."%(e2-s2))
    
    predictor.predictor(data, resultpath, protease)

####-----------------------------------------------####
if __name__ == '__main__':
    s0 = time.time()
    regular = '>(.*?)\s'
    protease = 'Trypsin'
    maxlen = 47
    minlen = 7
    missedcleavages = 2
    
    if len(sys.argv[1:]) <= 1:
        print("Error: wrong command! Please read User Guide of DeepDigest.\n"
              "Example: python the_main.py --input=input_filename "
              "--output=output_filename --regular=\">(.*?)\s\" "
              "--protease='Trypsin' --maxlen=47 --minlen=7 --missedcleavages=2")
        sys.exit(1)
    else:
        options, remainder = getopt.getopt(sys.argv[1:],'', 
                                           ['input=', 'output=', 'regular=', 
                                           'protease=', 'maxlen=', 'minlen=', 
                                           'missedcleavages='])
        for opt, arg in options:
            if opt == '--input':
                datapath = arg
            elif opt == '--output':
                resultpath = arg
            elif opt == '--regular':
                regular = arg
            elif opt == '--protease':
                protease = str(arg)
            elif opt == '--maxlen':
                maxlen = int(arg)
            elif opt == '--minlen':
                minlen = int(arg)
            elif opt == '--missedcleavages':
                missedcleavages = int(arg)
            else:
                print ("Error: Command-line argument: %s not recognized.\n"
                       "Exiting..." % opt)
                sys.exit()
    
    AI_tool(datapath, resultpath, regular, protease, 
            maxlen, minlen, missedcleavages)
    e0 = time.time()
    print("Time cost of the program was %s seconds."%(e0-s0))
    
    end = time.time()
    print("Time cost was %s seconds."%(end-start))
    
    print("----The program has finished running----")
