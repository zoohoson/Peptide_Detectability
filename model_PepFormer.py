from cgi import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda",0)


def genData(file,max_len):
    aa_dict={'A':1,'R':2,'N':3,'D':4,'C':5,'Q':6,'E':7,'G':8,'H':9,'I':10,
             'L':11,'K':12,'M':13,'F':14,'P':15,'O':16,'S':17,'U':18,'T':19,
             'W':20,'Y':21,'V':22,'X':23}
##############################################
# original code
    # with open(file, 'r') as inf:
    #     lines = inf.read().splitlines()
##############################################        
    lines = file
    long_pep_counter=0
    pep_codes=[]
    labels=[]
    lines.append('X'*79+',0')
    for pep in lines:
        pep,label=pep.split(",")
        labels.append(int(label))
        if not len(pep) > max_len:
            current_pep=[]
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > {}:".format(max_len),long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)  # padding
    return data, torch.tensor(labels)


class newModel(nn.Module):
    def __init__(self, vocab_size=24):
        super().__init__()
        self.hidden_dim = 25
        self.batch_size = 256
        self.emb_dim = 512
        
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2, 
                               bidirectional=True, dropout=0.2)
        
        
        self.block1=nn.Sequential(nn.Linear(2100,1024),
                                            nn.BatchNorm1d(1024),
                                            nn.LeakyReLU(),
                                            nn.Linear(1024,256),
                                 )

        self.block2=nn.Sequential(
                                               nn.BatchNorm1d(256),
                                               nn.LeakyReLU(),
                                               nn.Linear(256,128),
                                               nn.BatchNorm1d(128),
                                               nn.LeakyReLU(),
                                               nn.Linear(128,64),
                                               nn.BatchNorm1d(64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64,2)
                                            )
        
    def forward(self, x):
        x=self.embedding(x)
        output=self.transformer_encoder(x).permute(1, 0, 2)
        output,hn=self.gru(output)
        output=output.permute(1,0,2)
        hn=hn.permute(1,0,2)
        output=output.reshape(output.shape[0],-1)
        hn=hn.reshape(output.shape[0],-1)
        output=torch.cat([output,hn],1)
        return self.block1(output)

    def trainModel(self, x):
        with torch.no_grad():
            output=self.forward(x)
        return self.block2(output)


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        
        return loss_contrastive
    
    
def collate(batch):
    seq1_ls=[]
    seq2_ls=[]
    label1_ls=[]
    label2_ls=[]
    label_ls=[]
    batch_size=len(batch)
    for i in range(int(batch_size/2)):
        seq1,label1=batch[i][0],batch[i][1]
        seq2,label2=batch[i+int(batch_size/2)][0],batch[i+int(batch_size/2)][1]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label=(label1^label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1=torch.cat(seq1_ls).to(device)
    seq2=torch.cat(seq2_ls).to(device)
    label=torch.cat(label_ls).to(device)
    label1=torch.cat(label1_ls).to(device)
    label2=torch.cat(label2_ls).to(device)
    return seq1,seq2,label,label1,label2
    

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        x,y=x.to(device),y.to(device)
        outputs=net.trainModel(x)
        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def to_log(log):
    with open("log/PepFormer.log","a+") as f:
        f.write(log+'\n')


def main():
    train = pd.read_csv('data/train_diff_hpp.csv')[['peptide', 'label']].drop_duplicates()
    train['label'] = train['label'].apply(lambda x: {True: str(1), False: str(0)}[x])
    tmp1, tmp2 = train_test_split(train, test_size=0.2, random_state=7)
    train = []
    for line in tmp1.values:
        line = ','.join(line)
        train.append(line)
    val = []
    for line in tmp2.values:
        line = ','.join(line)
        val.append(line)
    tmp = pd.read_csv('data/test_diff_hpp.csv')[['peptide', 'label']].drop_duplicates()
    tmp['label'] = tmp['label'].apply(lambda x: {True: str(1), False: str(0)}[x])
    test = []
    for line in tmp.values:
        line = ','.join(line)
        test.append(line)
    
    train_data, train_label = genData(train, 81)
    train_dataset = Data.TensorDataset(train_data, train_label)
    val_data, val_label = genData(val, 81)
    val_dataset = Data.TensorDataset(val_data, val_label)
    test_data, test_label = genData(test, 81)
    test_dataset = Data.TensorDataset(test_data, test_label)

    batch_size=256
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                                    shuffle=True, collate_fn=collate)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    for num_model in range(1):  # just one train
        net=newModel().to(device)
        lr = 0.0001
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
        criterion = ContrastiveLoss()
        criterion_model = nn.CrossEntropyLoss(reduction='sum')
        
        best_acc=0
        EPOCH=250
        for epoch in range(EPOCH):
            loss_ls=[]
            loss1_ls=[]
            loss2_3_ls=[]
            t0=time.time()
            net.train()
            for seq1,seq2,label,label1,label2 in train_iter_cont:
                output1=net(seq1)
                output2=net(seq2)
                output3=net.trainModel(seq1)
                output4=net.trainModel(seq2)

                loss1=criterion(output1, output2, label)
                loss2=criterion_model(output3,label1)
                loss3=criterion_model(output4,label2)
                loss=loss1+loss2+loss3
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                loss_ls.append(loss.item())
                loss1_ls.append(loss1.item())
                loss2_3_ls.append((loss2+loss3).item())


            net.eval() 
            with torch.no_grad(): 
                train_acc=evaluate_accuracy(train_iter,net)
                test_acc=evaluate_accuracy(val_iter,net)
                
            results=f"epoch: {epoch+1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
            results+=f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc,"red")}, time: {time.time()-t0:.2f}'
            print(results)
            to_log(results)
            if test_acc>best_acc:
                best_acc=test_acc
                torch.save({"best_acc":best_acc,"model":net.state_dict()},f'log/PepFormer.pl')
                print(f"best_acc: {best_acc}")


if __name__ == '__main__':
    main()