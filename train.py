#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
from torch import nn as nn
import torch.nn.functional as func
#预处理文件，不是库
import preprocess
import os

NUM_EPOCHS=50
BATCH_SIZE=32
VAL_SPLIT_RATIO=0.1
LR=2.5e-4
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphConv(nn.Module):
   def __init__(self,num_f=128,num_layer=10):
     super(GraphConv,self).__init__()
     self.embedding=nn.Embedding(53,53)
     self.li=nn.Linear(72,num_f)
     self.linear=nn.ModuleList([nn.Linear(num_f,num_f)]*num_layer)
     self.classifier=nn.Sequential(nn.Linear(num_f,100),nn.ReLU(),nn.Dropout(0.4),nn.Linear(100,2))

   def forward(self,x,Adjacent):
       tmp=torch.argmax(x[:,:,:53],dim=2)
       x[:,:,:53]=self.embedding(tmp)
       x=self.li(x.float())
       for ln in self.linear:
         x=ln(x)
         x=torch.einsum('ijk,ikm->ijm',Adjacent.float().clone(),x)
         x=func.relu(x)
       x=x.mean(1)
       x=x.view(x.size(0),-1)
       result=self.classifier(x)
       return result


#validation
def forward(epoch,model,data_loader,criteria):
    validation_loss=0.0;validation_acc=0.0
    with torch.no_grad():
        model.eval()
        print("Begin validation epoch {}.".format(epoch+1))
        for i,(data,labels) in enumerate(data_loader):
          data[0],data[1]=data[0].to(device).long(),data[1].to(device).long()
          labels=labels.to(device).long()
          outputs=model(data[0],data[1])
          loss=criteria(outputs,labels)
          validation_loss+=loss.item()*labels.size(0)
          predictions = torch.max(outputs.data, 1)[1]
          correct_counts = predictions.eq(labels.data.view_as(predictions))
          validation_acc+=torch.sum(correct_counts.float()).item()
    return validation_loss,validation_acc


def train_val(model,train_loader,validation_loader,train_num,validation_num,criteria,optimizer,epoch,device):
  for t in range(epoch):
   train_loss = 0.0;train_acc = 0.0
   model.train()
   print("Begin train epoch {}.".format(t+1))
   for i,(data,labels) in enumerate(train_loader):
     data[0],data[1]=data[0].long().to(device),data[1].long().to(device)
     labels=labels.long().to(device)
     y_pred=model(data[0],data[1])
     loss=criteria(y_pred,labels)
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     train_loss+=loss.item()*labels.size(0)
     predictions=torch.max(y_pred.data,1)[1]
     correct= predictions.eq(labels.data.view_as(predictions))
     acc = torch.mean(correct.float())
     train_acc+= acc.item() * labels.size(0)
   validation_loss,validation_acc=forward(epoch=t,model=model,data_loader=validation_loader,criteria=criteria)
   avg_train_loss=train_loss/train_num
   avg_train_acc=train_acc/train_num
   avg_validation_loss=validation_loss/validation_num
   avg_validation_acc=validation_acc/validation_num
   print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}% \n\t\t Validation: Loss: {:.4f}, Accuracy: {:.4f}%".format(
        t + 1, avg_train_loss, avg_train_acc * 100,avg_validation_loss,avg_validation_acc * 100))
   if t+1 == NUM_EPOCHS:
        torch.save(model.state_dict(), os.path.join('weights','val_acc_final.pt'.format(t + 1)))


if __name__=='__main__':
    train_file='./train/names_smiles.txt'
    trainlabelfile='./train/names_labels.txt'
    validation_file='./validation/names_smiles.txt'
    validationlabelfile='./validation/names_labels.txt'

    dataset,label=preprocess.read_in(train_file,trainlabelfile,validation_file,validationlabelfile)
    training,training_label,validation,validation_label=preprocess.data_augmentation(dataset,label,split_ratio=VAL_SPLIT_RATIO)
    train_loader,validation_loader,train_num,validation_num=preprocess.generate_dataloader(training,training_label,validation,validation_label,batch_size=32,workers=4)
    print("Dataset load done.\nBatch size: {}. Total epoch: {}. Train dataset size: {}. Validation dataset size: {}. Val Split ratio: {}.".format(
            BATCH_SIZE,NUM_EPOCHS,train_num,validation_num, VAL_SPLIT_RATIO))

    model=GraphConv(128,3)
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    model=model.to(device)
    print(train_num)
    print(validation_num)
    train_val(model=model,train_loader=train_loader,validation_loader=validation_loader,
    train_num=train_num,validation_num=validation_num,criteria=loss_fn,optimizer=optimizer,epoch=NUM_EPOCHS,device=device)





