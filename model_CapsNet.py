from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import spacy
import sys
import argparse

from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data, datasets  # .legacy
from torchtext.vocab import Vectors
from torchsummary import summary
from torch.autograd import Variable

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def tokenizer(text): # create a tokenizer function
    """
    定义分词操作
    """
    return list(text)


def squash(x):
    epsilon = 0.00000001
    # not concern batch_size, maybe rewrite
    s_squared_norm = torch.sum(x*x,1,keepdim=True) + epsilon
    scale = torch.sqrt(s_squared_norm)/(1. + s_squared_norm)
    # out = (batch_size,1,10)*(batch_size,16,10) = (batch_size,16,10)
    out = scale * x
    return out


class Capsule(nn.Module):
    def __init__(self, in_units,in_channels, num_capsule, dim_capsule, routings=3, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.W = nn.Parameter((torch.randn(self.in_units,self.num_capsule,self.in_channels, self.dim_capsule)))

    def forward(self, u_vecs):
        u_vecs = u_vecs.permute(0,2,1)
        u_vecs = u_vecs.unsqueeze(2)
        u_vecs = u_vecs.unsqueeze(2)

        u_hat_vecs = torch.matmul(u_vecs,self.W)
        u_hat_vecs = u_hat_vecs.permute(0,1,2,4,3).squeeze(4)
        
        u_hat_vecs2 = u_hat_vecs.permute(0,2,1,3)
    
        b = torch.zeros(u_hat_vecs.size(0),self.num_capsule,1,self.in_units,device=DEVICE)
        for i in range(self.routings):
            c = F.softmax(b,-1)
            s = torch.matmul(c,u_hat_vecs2)
            s = s.permute(0,3,1,2).squeeze(3)
            v = squash(s)
            v = v.permute(0,2,1).unsqueeze(3)
            sim = torch.matmul(u_hat_vecs2,v)
            sim = sim.permute(0,1,3,2)
            b = b+sim
        return v.permute(0,2,1,3).squeeze(3)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    

def RECMEncoding(inpStr):
    RECMT=[]
    for x in inpStr:
        if x in RE:
            oneTi = energy[RE.get(x)]
            RECMT.append(oneTi)
    return RECMT
    

def RECMcompositionEncoding(inpStr):
    RECMcomposition=[]
    countNum = {'A':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
    for i in inpStr:
        if i in countNum:
            value = countNum.get(i)+1
            countNum[i] = value
    for i in countNum:
        oneTi = np.array(energy[RE.get(i)])*int(countNum.get(i))
        RECMcomposition.append(oneTi)
    RECMcomposition = np.array(RECMcomposition)
    return RECMcomposition
      

def GetPseRECM(RECMT):
    feature =[]
    legth = 0
    r = 3
    legth = 20+20*(r-1)
    # 取平均特征
    for j in range(20):
        averageColumn = 0
        for i in range(len(RECMT)):
            averageColumn = averageColumn + RECMT[i][j]
        averageColumn = averageColumn/len(RECMT)
        feature.append(averageColumn)
    for k in range(1,r):
        for j in range(20):
            dist = 0
            for i in range(len(RECMT)-k):
                dist = dist +pow((RECMT[i][j]-RECMT[i+k][j]),2)
            dist = dist/(len(RECMT)-k)
            feature.append(dist)
    feature = np.array(feature)
    return feature


def residueRatio(inpStr):
    feature = []
    countNum = {'A':0,'C':0,'D':0,'E':0,'F':0,'G':0,'H':0,'I':0,'K':0,'L':0,'M':0,'N':0,'P':0,'Q':0,'R':0,'S':0,'T':0,'V':0,'W':0,'Y':0}
    total = 0
    for i in inpStr:
        total = total+1
        if i in countNum:
            value = countNum.get(i)+1
            countNum[i] = value
    for i in countNum:
        oneResidueRatio = countNum.get(i)#/total
        feature.append(oneResidueRatio)
    feature = np.array(feature)
    return feature


def dipeptideRatio(inpStr):
    dipeptideFeature = np.zeros((20,20))
    total = 0
    for i in range(len(inpStr)-1):
        total = total+1
        x = RE.get(inpStr[i])
        y = RE.get(inpStr[i+1])
        dipeptideFeature[x][y] = dipeptideFeature[x][y] +1
    return dipeptideFeature


def physicochemicalFeature(inpStr,fixlength):
    pfeature=[]
    Slength = 0
    fix = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for x in inpStr:
        if x in RE:
            Slength = Slength+1
            oneTi = physicochemical[RE.get(x)]
            pfeature.append(oneTi)
    fixlength = fixlength-Slength
    for i in range(fixlength):
        pfeature.append(fix)
    pfeature = np.array(pfeature)
    return pfeature


def featureGenera(t):
    flag =0 
    for i in t:
        protein = ''   
        flag = flag+1
        for j in i:
            if j!=1:
                a = TEXT.vocab.itos[j]
                protein = protein+a
        featureOne = RECMcompositionEncoding(protein)
        featureTwo = GetPseRECM(RECMEncoding(protein))
        featureThree = dipeptideRatio(protein)
        featureFour = residueRatio(protein)
        featureFive = physicochemicalFeature(protein,45)
        featureTwo =featureTwo.reshape(3,20)
        featureFour = featureFour.reshape(1,20)
        featureOne = torch.from_numpy(featureOne)
        featureTwo = torch.from_numpy(featureTwo)
        featureThree = torch.from_numpy(featureThree)
        featureFour = torch.from_numpy(featureFour)
        featureFive = torch.from_numpy(featureFive)
        featureThree.type_as(featureTwo)
        featureFour.type_as(featureTwo)
        featureFive.type_as(featureTwo)
        feature1 = torch.cat((featureOne,featureTwo),0)
        feature2 = torch.cat((featureThree,featureFour.type_as(featureThree)),0)
        feature3 = torch.cat((feature1,feature2),0)
        feature = torch.cat((feature3,featureFive),0)
        if(flag==1):
            feature = feature.unsqueeze(0)
            temp = feature
        if(flag!=1):
            feature = feature.unsqueeze(0)
            temp = torch.cat((temp,feature),0)
    return temp


class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding = nn.Embedding(len_vocab,embed_size)
        self.lstm = nn.LSTM(embed_size,64,batch_first=True)#,bidirectional=True)
        self.conv = nn.Conv1d(embed_size,32,3)
        self.pool = nn.MaxPool1d(32)
        self.linear = nn.Linear(64,n_class)
        

    def forward(self, x):
        batch_size,seq_num = x.shape
        y = featureGenera(x)
        vec = self.embedding(x)
        out, (hn, cn) = self.lstm(vec)
        out = F.relu(out)
        out = self.linear(out[:,-1,:])
        out = F.softmax(out,-1)
        return out


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel,int(channel//reduction),bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction),channel,bias=False),
                                                )
        self.sigmoid = nn.Sigmoid()
        self.spatial_excitation = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7,
                                                 stride=1, padding=3, bias=False),
                                               )


    def forward(self, x):
        bahs, chs, _, _ = x.size()
        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_avg = self.avg_pool(x).view(bahs, chs)
        chn_avg = self.channel_excitation(chn_avg).view(bahs, chs, 1, 1)
        chn_max = self.max_pool(x).view(bahs, chs)
        chn_max = self.channel_excitation(chn_max).view(bahs, chs, 1, 1)
        chn_add=chn_avg+chn_max
        chn_add=self.sigmoid(chn_add)

        chn_cbam = torch.mul(x, chn_add)

        avg_out = torch.mean(chn_cbam, dim=1, keepdim=True)
        max_out, _ = torch.max(chn_cbam, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        spa_add = self.spatial_excitation(cat)
        spa_add=self.sigmoid(spa_add)

        spa_cbam = torch.mul(chn_cbam, spa_add)
        return spa_cbam


class CapsuleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len_vocab,embed_size)
        self.lstm = nn.LSTM(embed_size,40,batch_first=True)#,bidirectional=True)
        self.conv1=nn.Conv2d(1,256,9)
        self.conv3=nn.Conv2d(1,256,8)
        self.cbamBlock = CBAMBlock(256) 
        self.conv2=nn.Conv2d(256,32*8,9,2)
        self.conv4=nn.Conv2d(256,32*8,8,2) 
        self.capsule = Capsule(2304,16,2,32)

   
    def forward(self,x):
        batch_size = x.size(0)
        # Conv1
        y = featureGenera(x)
        out = self.embedding(x)
        y = y.type_as(out)
        out = out.unsqueeze(1)
        y = y.unsqueeze(1)
        out = self.conv1(out)
        out = self.cbamBlock(out)
        y = self.conv3(y)
        y = self.cbamBlock(y)
        out = F.relu(out)
        y = F.relu(y)

        out = self.conv2(out)
        y = self.conv4(y)
        out = F.relu(out)
        y = F.relu(y)
        out = out.view(batch_size,16,-1)
        y = y.view(batch_size,16,-1)
        out = torch.cat((out,y),2)
        out = squash(out)
        out = out.view(out.size(0),out.size(1),-1)
        out = self.capsule(out)
        out = out.permute(0,2,1)
        return out


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数。
这个映射被称为vocab。如果一个field已经被数字化了并且不需要被序列化，
可以将参数设置为use_vocab=False以及sequential=False。
"""
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer ,fix_length=45)
embed_size=20
n_class=2
n_hidden =32
energy = [
[-1.65,  -2.83, 1.16,	1.80,	-3.73,	-0.41,	1.90,	-3.69,	0.49,	-3.01,	-2.08,	0.66,	1.54,	1.20,	0.98, -0.08,  0.46, -2.31,	0.32,	-4.62],
[-2.83,	-39.58,	-0.82,	-0.53,	-3.07,	-2.96,	-4.98,	0.34,	-1.38,	-2.15,	1.43,	-4.18,	-2.13,	-2.91,	-0.41,	-2.33,	-1.84,	-0.16,	4.26,	-4.46],
[1.16,	-0.82,	0.84,	1.97,	-0.92,	0.88,	-1.07,	0.68,	-1.93,	0.23,	0.61,	0.32,	3.31,	2.67,	-2.02,	0.91,	-0.65,	0.94,	-0.71,	0.90],
[1.80,	-0.53,	1.97,	1.45,	0.94,	1.31,	0.61,	1.30,	-2.51,	1.14,	2.53,	0.20,	1.44,	0.10,	-3.13,	0.81,	1.54,	0.12,	-1.07,	1.29],
[-3.73,	-3.07,	-0.92,	0.94,	-11.25,	0.35,	-3.57,	-5.88,	-0.82,	-8.59,	-5.34,	0.73,	0.32,	0.77,	-0.40,	-2.22,	0.11,	-7.05,	-7.09,	-8.80],
[-0.41,	-2.96,	0.88,	1.31,	0.35,	-0.20,	1.09,	-0.65,	-0.16,	-0.55,	-0.52,	-0.32,	2.25,	1.11,	0.84,	0.71,	0.59,	-0.38,	1.69,	-1.90],
[1.90,	-4.98,	-1.07,	0.61,	-3.57,	1.09,	1.97,	-0.71,	2.89,	-0.86,	-0.75,	1.84,	0.35,	2.64,	2.05,	0.82,	-0.01,	0.27,	-7.58,	-3.20],
[-3.69,	0.34,	0.68,	1.30,	-5.88,	-0.65,	-0.71,	-6.74,	-0.01,	-9.01,	-3.62,	-0.07,	0.12,	-0.18,	0.19,	-0.15,	0.63,	-6.54,	-3.78,	-5.26],
[0.49,	-1.38,	-1.93,	-2.51,	-0.82,	-0.16,	2.89,	-0.01,	1.24,	0.49,	1.61,	1.12,	0.51,	0.43,	2.34,	0.19,	-1.11,	0.19,	0.02,	-1.19],
[-3.01,	-2.15,	0.23,	1.14,	-8.59,	-0.55,	-0.86,	-9.01,	0.49,	-6.37,	-2.88,	0.97,	1.81,	-0.58,	-0.60,	-0.41,	0.72,	-5.43,	-8.31,	-4.90],
[-2.08,	1.43,	0.61,	2.53,	-5.34,	-0.52,	-0.75,	-3.62,	1.61,	-2.88,	-6.49,	0.21,	0.75,	1.90,	2.09,	1.39,	0.63,	-2.59,	-6.88,	-9.73],
[0.66,	-4.18,	0.32,	0.20,	0.73,	-0.32,	1.84,	-0.07,	1.12,	0.97,	0.21,	0.61,	1.15,	1.28,	1.08,	0.29,	0.46,	0.93,	-0.74,	0.93],
[1.54,	-2.13,	3.31,	1.44,	0.32,	2.25,	0.35,	0.12,	0.51,	1.81,	0.75,	1.15,	-0.42,	2.97,	1.06,	1.12,	1.65,	0.38,	-2.06,	-2.09],
[1.20,	-2.91,	2.67,	0.10,	0.77,	1.11,	2.64,	-0.18,	0.43,	-0.58,	1.90,	1.28,	2.97,	-1.54,	0.91,	0.85,	-0.07,	-1.91,	-0.76,	0.01],
[0.98,	-0.41,	-2.02,	-3.13,	-0.40,	0.84,	2.05,	0.19,	2.34,	-0.60,	2.09,	1.08,	1.06,	0.91,	0.21,	0.95,	0.98,	0.08,	-5.89,	0.36],
[-0.08,	-2.33,	0.91,	0.81,	-2.22,	0.71,	0.82,	-0.15,	0.19,	-0.41,	1.39,	0.29,	1.12,	0.85,	0.95,	-0.48,	-0.06,	0.13,	-3.03,	-0.82],
[0.46,	-1.84,	-0.65,	1.54,	0.11,	0.59,	-0.01,	0.63,	-1.11,	0.72,	0.63,	0.46,	1.65,	-0.07,	0.98,	-0.06,	-0.96,	1.14,	-0.65,	-0.37],
[-2.31,	-0.16,	0.94,	0.12,	-7.05,	-0.38,	0.27,	-6.54,	0.19,	-5.43,	-2.59,	0.93,	0.38,	-1.91,	0.08,	0.13,	1.14,	-4.82,	-2.13,	-3.59],
[0.32,	4.26,	-0.71,	-1.07,	-7.09,	1.69,	-7.58,	-3.78,	0.02,	-8.31,	-6.88,	-0.74,	-2.06,	-0.76,	-5.89,	-3.03,	-0.65,	-2.13,	-1.73,	-12.39],
[-4.62,	-4.46,	0.90,	1.29,	-8.80,	-1.90,	-3.20,	-5.26,	-1.19,	-4.90,	-9.73,	0.93,	-2.09,	0.01,	0.36,	-0.82,	-0.37,	-3.59,	-12.39,	-2.68],
]
physicochemical = [
    [-0.4, -0.5, 15, 8.1, 0.046, 0.67, 1.28, 0.3, 0, 0.687, 115, 0.28, 154.330012, 27.5, 1.181, 0.0072,0,0,0,0],
    [0.17, -1, 47, 5.5, 0.128, 0.38, 1.77, 0.9, 2.75, 0.263, 135, 0.28, 219.789, 44.6, 1.461, -0.037,0,0,0,0],
    [-1.31, 3.0, 59, 13.0, 0.105, -1.2, 1.6, -0.6, 1.38, 0.632, 150, 0.21, 194.910002, 40.0, 1.587, 0.0238,0,0,0,0],
    [-1.22, 3.0, 73, 12.3, 0.151, -0.76, 1.56, -0.7, 0.92, 0.669, 190, 0.33, 223.160, 62, 1.862, 0.0068,0,0,0,0],
    [1.92, -2.5, 91, 5.2, 0.29, 2.3, 2.94, 0.5, 0, 0.577, 210, 2.18, 204.7, 115.5, 2.228, 0.0376,0,0,0,0],
    [-0.67, 0, 1, 9, 0, 0, 0, 0.3, 0.74, 0.67, 75, 0.18, 127.9, 0, 0.881, 0.179,0,0,0,0],
    [-0.64, -0.5, 82, 10.4, 0.23, 0.64, 2.99, -0.1, 0.58, 0.594, 195, 0.21, 242.539, 79, 2.025, -0.011,0,0,0,0],
    [1.25, -1.5, 57, 5.2, 0.186, 1.9, 4.19, 0.7, 0, 0.564, 175, 0.82, 233.210, 93.5, 1.81, 0.0216,0,0,0,0],
    [-0.67, 3, 73, 11.3, 0.219, -0.57, 1.89, -1.8, 0.33, 0.407, 200, 0.09, 300.459, 100, 2.258, 0.0177,0,0,0,0],
    [1.22, -1.8, 57, 4.9, 0.186, 1.9, 2.59, 0.5, 0, 0.541, 170, 1, 232.3, 93.5, 1.931, 0.0517,0,0,0,0],
    [1.02, -1.3, 75, 5.7, 0.0221, 2.4, 2.35, 0.4, 0, 0.328, 185, 0.74, 202.699, 94.1, 2.034, 0.0027,0,0,0,0],
    [-0.92, 0.2, 58, 11.6, 0.134, -0.61, 1.6, -0.5, 1.33, 0.489, 160, 0.25, 207.899, 58.7, 1.655, 0.0054,0,0,0,0],
    [-0.49, 0, 42, 8.0, 0.131, 102, 2.67, -0.3, 0.39, 0.600, 145, 0.39, 179.929, 41.9, 1.468, 0.239,0,0,0,0],
    [-0.91, 0.2, 72, 10.5, 0.180, -0.22, 1.56, -0.7, 0.9, 0.527, 183, 0.35, 235.509, 80.7, 1.932, 0.0692,0,0,0,0],
    [-0.59, 3, 101, 10.5, 0.291, -2.10, 2.34, -1.4, 0.64, 0.591, 225, 0.1, 341.0, 105, 2.56, 0.0436,0,0,0,0],
    [-0.55, 0.3, 31, 9.2, 0.062, 0.01, 1.31, -0.1, 1.41, 0.693, 116, 0.12, 174.059, 29.3, 1.298, 0.0043,0,0,0,0],
    [-0.28, -0.4, 45, 8.6, 0.108, 0.52, 3.03, -0.2, 0.71, 0.713, 142, 0.21, 205.5, 51.3, 1.525, 0.034,0,0,0,0],
    [0.91, -1.5, 43, 5.9, 0.14, 1.5, 3.67, 0.6, 0, 0.529, 157, 0.6, 207, 71.5, 1.645, 0.057,0,0,0,0],
    [0.5, -3.4, 130, 5.4, 0.409, 2.6, 3.21, 0.3, 0.12, 0.632, 258, 5.7, 237, 145.5, 2.663, 0.058,0,0,0,0],
    [1.67, -2.3, 107, 6.2, 0.298, 1.6, 2.94, -0.4, 0.21, 0.493, 234, 1.26, 229.14, 117.3, 2.368, 0.0236,0,0,0,0]
]
RE = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}


# train test split
train = pd.read_csv('data/train_diff_hpp.csv')[['peptide', 'label']].drop_duplicates().rename({'peptide': 'Seq','label': 'Label'}, axis=1)
train['Label'] = train.Label.apply(lambda x: {True: 1, False:0}[x])
test = pd.read_csv('data/test_diff_hpp.csv')[['peptide', 'label']].drop_duplicates().rename({'peptide': 'Seq','label': 'Label'}, axis=1)
test['Label'] = test.Label.apply(lambda x: {True: 1, False:0}[x])
train, val = train_test_split(train, test_size=0.2, random_state=7)
train.to_csv('2021MDPI_CapsNet/train.csv', index=False)
val.to_csv('2021MDPI_CapsNet/val.csv', index=False)
test.to_csv('2021MDPI_CapsNet/test.csv', index=False)
train, val, test = data.TabularDataset.splits(
                                path='2021MDPI_CapsNet/', 
                                train='train.csv',
                                validation='val.csv',
                                test='test.csv',
                                format='csv',
                                skip_header=True,
                                fields=[('Seqs', TEXT), ('Label', LABEL)])
TEXT.build_vocab(train,val,test)
len_vocab = len(TEXT.vocab)


train_iter = data.BucketIterator(train, batch_size=32, sort_key=lambda x: len(x.Seqs), 
                            shuffle=True,device=DEVICE)
val_iter = data.BucketIterator(val, batch_size=32, sort_key=lambda x: len(x.Seqs), 
                                shuffle=True,device=DEVICE)
test_iter = data.BucketIterator(test, batch_size=32, sort_key=lambda x: len(x.Seqs), 
                                 shuffle=True,device=DEVICE)
# modeling
model = CapsuleNet()
# 训练
model.to(DEVICE)
# 训练
optimizer = optim.Adam(model.parameters(),lr=0.001)
n_epoch = 12
best_val_acc = 0

#train_iter
for epoch in range(n_epoch):
    for batch_idx, batch in enumerate(train_iter):
        data = batch.Seqs
        target = batch.Label
        target = torch.sparse.torch.eye(n_class).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1,0)
        optimizer.zero_grad()
        output = model(data)
        output = torch.sqrt(torch.sum(output*output, 2))
        loss1 = target*F.relu(0.9-output)**2 + 0.5*(1-target)*F.relu(output-0.1)**2
        loss1 = loss1.sum(dim=1).mean()
        loss = loss1#+0.0005*loss2
        loss.backward()
        optimizer.step()

        if (batch_idx+1) %100 == 0:
            _,y_pre = torch.max(output,-1)
            acc = torch.mean((torch.tensor(y_pre == batch.Label,dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                %(epoch,batch_idx,loss,acc))
    
    val_accs = []
    #val_iter
    for batch_idx, batch in enumerate(val_iter):
        data = batch.Seqs
        target = batch.Label
        target = torch.sparse.torch.eye(n_class).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1,0)
        out = model(data)
        out = torch.sqrt(torch.sum(out*out, 2))
        _,y_pre = torch.max(out,-1)
        acc = torch.mean((torch.tensor(y_pre == batch.Label,dtype=torch.float)))
        val_accs.append(acc)
    acc =torch.mean(torch.stack(val_accs))
    if acc > best_val_acc:
        print('val acc : %.4f > %.4f saving model'%(acc,best_val_acc))
        torch.save(model.state_dict(), 'log/CapsNet.pkl')
        best_val_acc = acc
    print('test acc: %.4f'%(acc))



# # test
# all_pred1 = []
# all_true1 = []
# all_p1 = []
# for batch_idx, batch in enumerate(test_iter):
#     data = batch.Seqs
#     data = data.permute(1, 0)
    
#     out = model(data)
#     out = torch.sqrt(torch.sum(out * out, 2))
#     out1 = out[:, 1]
#     _, y_pre = torch.max(out, -1)
#     all_p1.extend(list(out1.cpu().detach().numpy()))
#     all_pred1.extend(list(y_pre.cpu().detach().numpy()))
#     all_true1.extend(list(batch.Label.cpu().detach().numpy()))



# auc = roc_auc_score(all_true1, all_p1)
# print(metrics.accuracy_score(all_true1,all_pred1))
# print('TEST AUC : {}'.format(auc))
# plt.figure(figsize=(4, 4))
# fpr, tpr, _ = roc_curve(all_true1, all_p1)
# plt.plot(fpr, tpr, marker='.', label='AUC = {:.4f}'.format(auc), color='orange')
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.savefig('log/CapsNet_roc.png')

# cf_matrix = confusion_matrix(all_true1, all_pred1)
# plt.figure(figsize=(3, 3))
# sns.heatmap(cf_matrix, annot=True, fmt=',.0f')
# plt.savefig('log/CapsNet_CM.png')
# plt.show()

# np.save('log/result_CapsNet_fpr.npy', fpr)
# np.save('log/result_CapsNet_tpr.npy', tpr)