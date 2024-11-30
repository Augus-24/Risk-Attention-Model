import pandas as pd
import numpy as np
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = './'


class DataProoducer():
    def __init__(self):
        self.tradingdays = []
        self.timeseries = []
        self.stocks = []

    def get_trade_dates(self, start_date, end_date=None):
        '''
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
        filename = f'{self.tradingdays[tidx]}.pkl'
        path = os.path.join(BASE_DIR,dmgr,filename)
        data = pd.read_pickle(path)
        return data  
    
    def fill_data(self,df,nan_pct):
        if len(df) <= 30*(1-nan_pct):
            return None
        
        trade_dt = pd.DataFrame({'TRADE_DT':self.timeseries})
        df['TRADE_DT'] = df['TRADE_DT'].astype(str)
        trade_dt['TRADE_DT'] = trade_dt['TRADE_DT'].astype(str)

        df = pd.merge(trade_dt,df,on='TRADE_DT',how='left')
        df.fillna(method='bfill',inplace=True)
        df.fillna(method='ffill',inplace=True)
        return df

    def run_OHLCV(self,dmgr,T0):
        '''
        :param dmgr: 文件地址   
        :param T0: 给定T日索引，下面循环向前取至T-29，共30天。
        :return: OHLCV数据
        '''
        fields = ['S_DQ_OPEN','S_DQ_HIGH','S_DQ_LOW','S_DQ_CLOSE','S_DQ_VOLUME']
        add = ['S_INFO_WINDCODE','TRADE_DT']
        data = pd.DataFrame()
        tidx = T0
        self.timeseries=[]
        for tidx in range(T0-29,T0+1):
            temp = self.get_data(dmgr,tidx)
            temp = temp[add+fields]
            data = pd.concat([data,temp],axis=0)
            self.timeseries.append(self.tradingdays[tidx])

        stocks_to_remove = ['000300.SH','000852.SH','000016.SH','000905.SH']
        data = data[~data.iloc[:, 0].isin(stocks_to_remove)]
        #标准化并保留8位小数
        data[fields] = ((data[fields]-data[fields].mean())/data[fields].std()).round(8)
        final = data.groupby('S_INFO_WINDCODE',as_index=False).apply(self.fill_data,0.5)
        return final
    
    def run_factors(self,dmgr,T0_day):
        '''
        标准化风险因子
        '''
        data = self.get_data(dmgr,T0_day)
        type_factors = ['beta', 'momentum', 'size', 'earnyild',
       'resvol', 'growth', 'btop', 'leverage', 'liquidty', 'sizenl']
        data[type_factors] = (data[type_factors]-data[type_factors].mean())/data[type_factors].std()
        return data
  
    def run_label(self,dmgr,T0_day):
        fields = ['S_INFO_WINDCODE','S_DQ_CLOSE']
        data1 = self.get_data(dmgr,T0_day+1)[fields]
        data1 = data1.rename(columns={'S_DQ_CLOSE':'close_1'})
        data2 = self.get_data(dmgr,T0_day+11)[fields]
        data2 = data2.rename(columns={'S_DQ_CLOSE':'close_11'})
        df = pd.merge(data1,data2,on='S_INFO_WINDCODE',how='inner')
        df['pct'] = (df['close_11']-df['close_1'])/df['close_1']
        df.drop(columns=['close_1','close_11'],axis=1,inplace=True)
        return df

    def run(self,dmgr1,dmgr2,T0):
        df1 = self.run_OHLCV(dmgr1,T0)
        df2 = self.run_factors(dmgr2,T0)
        labels = self.run_label(dmgr1,T0)
        self.stocks = list(set(df1['S_INFO_WINDCODE'])&set(df2['symbol'])&set(labels['S_INFO_WINDCODE']))
        df1 = df1[df1['S_INFO_WINDCODE'].isin(self.stocks)]
        df1 = df1.sort_values(['S_INFO_WINDCODE','TRADE_DT'],ascending=True)
        df2 = df2[df2['symbol'].isin(self.stocks)] 
        df2 = df2.sort_values(['symbol'],ascending=True)
        labels = labels[labels['S_INFO_WINDCODE'].isin(self.stocks)]
        labels = labels.sort_values(['S_INFO_WINDCODE'],ascending=True)
        fields = ['S_DQ_OPEN','S_DQ_HIGH','S_DQ_LOW','S_DQ_CLOSE','S_DQ_VOLUME']
        l = len(self.stocks)
        self.stocks = []
        res1 = df1[fields].values.reshape(l,30,5)
        df2.drop(columns=['trade_date','symbol'],inplace=True)
        res2 = df2.values.reshape(l,-1)
        res3 = labels['pct'].values.reshape(-1)
        day = self.tradingdays[T0]
        path1 = os.path.join('train_data/OHLCV',f'{day}.npy')
        path2 = os.path.join('train_data/factors',f'{day}.npy')
        path3 = os.path.join('train_data/labels',f'{day}.npy')
        np.save(path1,res1)
        np.save(path2,res2)
        np.save(path3,res3)
        
        

dmgr_OHLCV = 'eod_price'
dmgr_barra = 'barra/exposure'
producer = DataProoducer()
producer.get_trade_dates(start_date='20100801',end_date='20220701')
for tidx in range(29,len(producer.tradingdays)-12):
    producer.run(dmgr_OHLCV,dmgr_barra,tidx)
    print(f'finish: {producer.tradingdays[tidx]}')
    



    
    
