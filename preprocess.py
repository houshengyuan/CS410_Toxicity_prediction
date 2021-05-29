#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import torch.utils.data
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

atom_list=[ 'C','N','O','S','P','F','Cl','Br','B','H','Ca', 'Zr', 'Si', 'Dy', 'Pb', 'V', 'Ti','Sr',
           'Bi', 'Pd', 'Al', 'Yb', 'Mn', 'Tl', 'As', 'Mo', 'Fe',
           'Sn', 'Ru', 'K', 'Pt', 'Li', 'Ag', 'Au', 'Sb', 'Cd', 'Mg',
           'Cu', 'Cr', 'Be', 'Nd', 'Co', 'I', 'Ba', 'Ge', 'Eu',
            'Zn', 'In', 'Hg', 'Na', 'Se', 'Sc', 'Ni']
degree=[0,1,2,3,4,5,6]
total_H=[0,1,2,3,4]
valence=[0,1,2,3]

class my_dataset(torch.utils.data.Dataset):
   def __init__(self,smileslist,labellist):
     self.smiles=smileslist
     self.labels=list(map(int,labellist))
     self.max_natoms=132

   def __len__(self):
     return len(self.smiles)

   def atom_f(self,mole,id):
     atom=mole.GetAtomWithIdx(id)
     type=list(map(lambda s: atom.GetSymbol() == s,atom_list))
     get_degree=list(map(lambda s: atom.GetDegree()==s,degree))
     getH=list(map(lambda s: atom.GetTotalNumHs()==s,total_H))
     getvalence=list(map(lambda s: atom.GetImplicitValence()==s,valence))
     aromatic=[atom.GetIsAromatic()]
     hybridization=[atom.GetHybridization()]
     ring=[atom.IsInRing()]
     res=type+get_degree+getH+getvalence+aromatic+hybridization+ring
     return np.array(res).astype(float)

   def __getitem__(self, item):
      smile=Chem.MolFromSmiles(self.smiles[item])
      A_padding = np.zeros((self.max_natoms, self.max_natoms))
      lbl=torch.tensor(self.labels[item])
      if smile:
       natoms=smile.GetNumAtoms()
       A=GetAdjacencyMatrix(smile)+np.eye(natoms)
       A_padding[:natoms,:natoms] = A

       feature= [self.atom_f(smile,i) for i in range(natoms)]
       for i in range(self.max_natoms-natoms):
         feature.append(np.zeros(72))
       feature=torch.from_numpy(np.array(feature))
       return [(feature,A_padding),lbl]
      else:
        feature=[np.zeros(72) for i in range(self.max_natoms)]
        feature=torch.from_numpy(np.array(feature))
        return [(feature,A_padding),lbl]


def read_in_data(filename,islabel):
  file=open(filename,"r")
  line=file.readlines()
  data=[]
  for i,x in enumerate(line):
    if i:
     tmp=x.strip().split(',')[1]
     data.append(tmp)
  file.close()
  return data

#read in the one-hot matrix
def read_in(datasetfile,labelfile,validationsetfile,validationlabelfile):
  smiles1=read_in_data(datasetfile,0)
  label1=read_in_data(labelfile,1)
  smiles2=read_in_data(validationsetfile,0)
  label2=read_in_data(validationlabelfile,1)
  return smiles1+smiles2,label1+label2

def split(dataset,label,split_ratio=0.2,type="neg",contrast=1):
 np.random.seed(44)
 shuffled_indices=np.random.permutation(len(dataset))
 validation_set_size=int(len(dataset) * split_ratio)
 training=[];training_label=[]
 validation=[];validation_label=[]
 validation_indices=shuffled_indices[-validation_set_size:]
 train_indices=shuffled_indices[:-validation_set_size]
 rotten=[]
 for i,data in enumerate(dataset):
    if not Chem.MolFromSmiles(data):
        rotten.append(i)
 for i in train_indices:
   if i not in rotten:
    training.append(dataset[i])
    training_label.append(label[i])
 for i in validation_indices:
   if i not in rotten:
    validation.append(dataset[i])
    validation_label.append(label[i])
 if type=="pos":
  return training*contrast,training_label*contrast,validation,validation_label
 else: return training,training_label,validation,validation_label

def data_augmentation(dataset,label,split_ratio):
    negative_sample=[dataset[i] for i in range(len(dataset)) if label[i]=='0']
    negative_label=[label[i] for i in range(len(dataset)) if label[i]=='0']
    positive_sample=[dataset[i] for i in range(len(dataset)) if label[i]=='1']
    positive_label=[label[i] for i in range(len(dataset)) if label[i]=='1']
    contrast=len(negative_sample)//len(positive_sample)
    print(contrast)
    training1,training_label1,validation1,validation_label1=split(negative_sample,negative_label,split_ratio=split_ratio)
    training2,training_label2,validation2,validation_label2=split(positive_sample,positive_label,split_ratio=split_ratio,type="pos",contrast=7)
    return training1+training2,training_label1+training_label2,\
           validation1+validation2,validation_label1+validation_label2

def generate_dataloader(train,train_label,validation,validation_label,batch_size=32,workers=4):
   train_set = my_dataset(smileslist=train,labellist=train_label)
   validation_set=my_dataset(smileslist=validation,labellist=validation_label)
   train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
   validation_loader=torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=workers)
   return train_loader,validation_loader,len(train_set),len(validation_set)
