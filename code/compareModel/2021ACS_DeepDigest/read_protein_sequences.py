import os,sys,re

# read the FASTA file
def read_protein_sequences(file, regular):
    # check the file path
    if os.path.exists(file) == False:
        print("Error: The input file does not exist. Please check again.")
        sys.exit(1)
    # record protein names and sequences
    with open(file) as f:
        records = f.read()
    # FASTA file must start with character '>'
    if re.search('>', records) == None:
        print("Error: The input file seems not in FASTA format!")
        sys.exit(1)
    # check the regular expression
    elif re.search(regular, records) == None:
        print("Error: Cannot parse the fasta file by the regular expression.")
        sys.exit(1)
    
    records = re.split('(>.*?)\\n', records)[1:]
    
    l = len(records)
    fasta_sequences = []
    for i in range(0, l, 2):
        name = re.split(regular, records[i])[1]
        sequence = records[i + 1].replace('\n', '')
        fasta_sequences.append([name, sequence])
    return fasta_sequences
