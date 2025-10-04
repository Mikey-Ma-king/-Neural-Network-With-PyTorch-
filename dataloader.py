from torch.utils.data import Dataset

import os
import torch

import pandas as pd
import numpy as np

class iris_dataloader(Dataset):
    def __init__(self,datapath):
        super().__init__()

        self.datapath = datapath
        assert os.path.exists(self.datapath), "Datapath isn't exist"

        df = pd.read_csv(self.datapath,names = [1,2,3,4,5,6]) 

        data = df.iloc[:,0:5]

        label = df.iloc[:,5]

        self.data = torch.from_numpy(np.array(data,dtype = "float32"))
        self.label = torch.from_numpy(np.array(label,dtype = "int64"))

        self.num =  len(label)

        print("当前数据集的大小是：",self.num)
    
    def __len__(self):
        return self.num
    
    def __getitem__(self,index):
        return self.data[index] , self.label[index]










