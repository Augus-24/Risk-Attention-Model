import pandas as pd
import numpy as np
import os
from datetime import datetime,timedelta
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import warnings
from torch.nn.utils import clip_grad_norm_
import scipy.stats as stats
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
warnings.filterwarnings('ignore')


BASE_DIR = 'train_data'

class RiskAttention(nn.Module):
    def __init__(self, embed_size,output_size,input_size, hidden_size,num_layers,MLP_hidden):
        super(RiskAttention,self).__init__()
        self.embed_size = embed_size
        self.output_size = output_size
        self.keys_para = nn.Linear(self.embed_size,self.output_size,bias=True)
        self.queries_para = nn.Linear(self.embed_size,self.output_size,bias=True)
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.values_para = nn.Linear(hidden_size,self.output_size,bias=True)
        self.fc1 = nn.Linear(2*hidden_size,MLP_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(MLP_hidden,1)
    
    def forward(self,values,keys,queries):
        keys = self.keys_para(keys)
        queries = self.queries_para(queries)
        H, _ = self.gru(values)
        H = H[:,-1,:]
        values = self.values_para(H)
        N = queries.shape[0]
        L = values.shape[1]
        energy = torch.matmul(queries, keys.T)
        attention = nn.Softmax(dim=1)(energy/(L**(1/2)))
        output = torch.matmul(attention, values)
        final = torch.cat((output,values),dim=1)
        grades = self.fc1(final)
        grades = self.relu(grades)
        grades = self.fc2(grades)
        grades = grades.squeeze(-1)
        return grades


class MyLightningModule(pl.LightningModule):
    def __init__(self,chosen_size,learning_rate,size2):
        super(MyLightningModule, self).__init__()
        self.model = RiskAttention(embed_size=39,output_size=chosen_size,input_size=5,hidden_size=chosen_size,num_layers=1,MLP_hidden=size2)
        self.learning_rate = learning_rate


    def forward(self,OHLCV,factors):
        return self.model(OHLCV,factors,factors)
    
    def ccc_loss(self,y_pred,y_true):
        """
        计算CCC损失函数
        :param y_pred: 预测值
        :param y_true: 真实值
        :return: CCC损失值
        """
        mean_true = torch.mean(y_true)
        mean_pred = torch.mean(y_pred)
        var_true = torch.var(y_true)
        var_pred = torch.var(y_pred)
        cov_true_pred = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc = (2 * cov_true_pred) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return 1-ccc

    def training_step(self, batch, batch_idx):
        OHLCV,factors,y = batch
        idx = OHLCV.shape[0]
        OHLCV = OHLCV.squeeze(0)
        factors = factors.squeeze(0)
        y = y.squeeze(0)
        pred = self(OHLCV,factors)
        train_loss = self.ccc_loss(pred,y)
        self.log('train_loss',train_loss,on_step=False,on_epoch=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        OHLCV,factors,y_true = batch
        idx = OHLCV.shape[0]
        OHLCV = OHLCV.squeeze(0)
        factors = factors.squeeze(0)
        y_true = y_true.squeeze(0)

        y_pred = self(OHLCV,factors)
        loss = self.ccc_loss(y_pred,y_true)
        rank_pred1 = stats.rankdata(y_pred.detach().numpy())
        rank_label1 = stats.rankdata(y_true.detach().numpy())
        rank_IC = stats.spearmanr(rank_pred1,rank_label1)[0]
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('loss',loss,on_step=False,on_epoch=True)
        self.log('rankIC',rank_IC,on_step=False,on_epoch=True)
        self.log('learning_rate',round(current_lr,6),on_step=False,on_epoch=True)
        return loss,rank_IC

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler,'monitor': 'loss','interval': 'epoch'}


class GetData(Dataset):
    def __init__(self):
        super(GetData,self).__init__()
        self.tradingdays = []
    def get_trade_dates(self, start_date, end_date=None):
        '''
        获取交易日期
        :param start_date:
        :param end_date:
        :return:
        '''
        #tradingDateV = np.load("/data/data-base/calendar_wind.npy")
        if end_date == None:
            end_date = datetime.now().strftime('%Y%m%d')
        tradingDateV = np.load("./calendar_wind.npy") 
        tradingdays = tradingDateV[(tradingDateV>=int(start_date))&(tradingDateV<=int(end_date))]
        self.tradingdays = list(tradingdays)
        return None

    def get_data(self,dmgr,tidx):
        path = os.path.join(BASE_DIR,dmgr,f'{self.tradingdays[tidx]}.npy')
        data = np.load(path,allow_pickle=True)
        data = torch.Tensor(data)
        return data
    
    def __len__(self):
        return len(self.tradingdays)
    
    def __getitem__(self, index):
        OHLCV = self.get_data('OHLCV',index)
        factors = self.get_data('factors',index)
        label = self.get_data('labels',index)
        sample = (OHLCV,factors,label)
        return sample
    
train_set = GetData()
train_set.get_trade_dates('20080219','20180231')
train_loader = DataLoader(train_set,batch_size=1,shuffle=True)

test_set = GetData()
test_set.get_trade_dates('20180301','20190301')
test_loader = DataLoader(test_set,batch_size=1,shuffle=True)



#chosen_size = 32
chosen_size = 64
learning_rate_ = [0.00001]
size2_ = [128]

for size2 in size2_:
    for learning_rate in learning_rate_:
        name = f"{learning_rate}_{chosen_size}_{size2}_relu"
        logger = CSVLogger("./", name=name)
        model = MyLightningModule(chosen_size,learning_rate,size2)
        early_stopping = EarlyStopping(monitor='rankIC',patience=20,min_delta=0,mode='max')
        checkpoint_callback = ModelCheckpoint(
            monitor='rankIC',
            dirpath='./checkpoints/',
            filename='{epoch:02d}-{rankIC:.2f}',
            save_top_k=1,
            mode='max',
        )
        trainer = pl.Trainer(max_epochs=200,
                            callbacks=[early_stopping,checkpoint_callback],
                            logger=logger,
                            gradient_clip_val=3.0,
                            gradient_clip_algorithm='norm')
        trainer.fit(model,train_loader,test_loader)





