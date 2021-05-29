#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
#是训练python文件，不是库
import train
#是预处理文件，不是库
import preprocess
import torch.utils.data
import torch.nn.functional as func

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def forward(model,data_loader):
    res=[]
    with torch.no_grad():
        model.eval()
        for i,(data,labels) in enumerate(data_loader):
          data[0],data[1]=data[0].to(device).long(),data[1].to(device).long()
          outputs=model(data[0],data[1])
          output =func.softmax(outputs, dim=1)
          y_pred = list(output[:, 1])
          y_pred = [i.item() for i in y_pred]
          res+=y_pred
    return  res

if __name__ == "__main__":
    loaded_model_state=torch.load("./weights/val_acc_final.pt")
    loaded_model=train.GraphConv(128,3).to(device)
    loaded_model.load_state_dict(loaded_model_state)
    file=open("test/names_smiles.txt","r")
    content=file.readlines()
    data=[]
    for i,line in enumerate(content):
     if i:
       data.append(line.strip().split(',')[1])
    test_set = preprocess.my_dataset(smileslist=data,labellist=[0]*len(data))
    test_loader=torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)
    y_pred=forward(model=loaded_model,data_loader=test_loader)
    input=open("test/output_sample.txt", "r")
    output=open("test/output_518021910604.txt", "w")
    line = input.readlines()
    count = 0
    for lines in line:
        count += 1
        if lines[0] != 'C':
            tmp = lines.split(',')
            output.write(tmp[0] + ',' + str(round(y_pred[count-2],2)))
        else:
            output.write(lines)
        if count != len(line) and count!=1:
            output.write('\n')
