import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = './Task2/1e-05_64_128_relu/version_3'
lr = 0.00001
chosen_size = 64
method = 'ReLu'

data = pd.read_csv(os.path.join(base_path,'metrics.csv'))
data1= data[~data['rankIC'].isnull()][['epoch','loss','rankIC']]
train_loss = data[~data['train_loss'].isnull()]['train_loss']

epoch = data1['epoch']
fig,ax1 = plt.subplots()
ax1.set_xlabel(f'epoch:lr={lr},chosen_size={chosen_size},{method}')
ax1.set_ylabel('rankIC')
ax1.plot(epoch,data1['rankIC'],label='rankIC')

ax2 = ax1.twinx()
ax2.set_ylabel('loss')
ax2.plot(epoch,data1['loss'],'r',label='val_loss')
ax2.plot(epoch,train_loss,'purple',label='train_loss')

lines1,labels1 = ax1.get_legend_handles_labels()
lines2,labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2,labels1+labels2,loc='upper right')
fig.tight_layout()
plt.savefig(os.path.join(base_path,'train_result.png'))
