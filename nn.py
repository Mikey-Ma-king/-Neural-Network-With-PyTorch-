import torch.utils.data as u_data
import sys
import os

import torch
import torch.nn as nn
import torch.optim as op

from tqdm import tqdm

from dataloader import iris_dataloader as ir

class NN(nn.Module):
    def __init__(self,in_dim,hidden_dim1,hidden_dim2,out_dim):
       super().__init__()
       self.layer1 = nn.Linear(in_dim,hidden_dim1)
       self.layer2 = nn.Linear(hidden_dim1,hidden_dim2)
       self.layer3 = nn.Linear(hidden_dim2,out_dim)

    def forward(self,x):
        x = torch.relu((self.layer1(x)))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  

        return x 
    
device = torch.device("cpu")

custom_dataset =ir("D:/Python/Neural Network/ir.csv")
train_size = int(custom_dataset.num*0.6)
val_size  = int(custom_dataset.num*0.2)
test_size  = int(custom_dataset.num*0.2)

train_dataset,val_dataset,test_dataset= u_data.random_split(custom_dataset,[train_size,val_size,test_size])

train_loader = u_data.DataLoader(train_dataset,batch_size=8,shuffle=True)
val_loader = u_data.DataLoader(val_dataset,batch_size=1,shuffle=False)
test_loader = u_data.DataLoader(test_dataset,batch_size=1,shuffle=False)

print("训练集的大小是：",train_size,'\n',"验证集的大小是：",val_size,'\n',"测试集的大小是：",test_size,'\n')

def infer(model,dataset,device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas,label = data
            outputs = model(datas.to(device))
            predictions = torch.argmax(outputs,dim = 1)
            acc_num += torch.eq(predictions,label.to(device)).sum().item()
    acc = acc_num/len(dataset)
    return acc

def main(lea_rate = 0.005,turns = 200):
    model = NN(5,12,6,3).to(device)
    loss_f = nn.CrossEntropyLoss()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = op.Adam(pg,lea_rate)

    save_path = os.path.join(os.getcwd(),"results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for turn in range(turns) :
        model.train()
        #sample_num = 0

        train_bar = tqdm(train_loader, file = sys.stdout, ncols = 100)#显示训练过程，可视化

        for datas in train_bar:
            data,label = datas
            #sample_num += len(data)

            optimizer.zero_grad()
            outputs = model(data.to(device))

            loss = loss_f(outputs,label.to(device))
            loss.backward()
            optimizer.step()

            train_bar.desc = "train turns {}/{} loss {:.4f}".format(turn+1,turns,loss)

        val_acc = infer(model,val_loader,device)
        print("val turn {}/{}  val_acc {}".format(turn+1,turns,val_acc))
        torch.save(model.state_dict(),os.path.join(save_path,"nn.pth"))
        val_acc = 0
        
    print("Finish Training")

    test_acc = infer(model,test_loader,device)
    print("test_acc = ",test_acc)

if __name__ =="__main__" :
    main()   
    










