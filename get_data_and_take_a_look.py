# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
big_block = pd.read_csv('/home/lin/教材/Financial_data/data/finance_M4/Daily-train.csv')
big_block.head()
big_block.set_index('V1', inplace = True)
fx_data = big_block.loc['D3450']
fx_data.head()
lastNumber = fx_data.last_valid_index()
lastNumber
my_data = fx_data.loc[fx_data.isnull().values == False]
my_data.head()
my_data.tail()
my_data.size
type(my_data)
my_data.to_csv('/home/lin/教材/Financial_data/data/finance_M4/my_fx_data')
my_data.describe()
my_data.dtypes
M4 = pd.read_csv('/home/lin/教材/Financial_data/data/finance_M4/M4-info.csv')
M4.head()
M4.set_index('M4id', inplace = True)
M4.loc['D3450']
my_data.plot()
logR = np.log(my_data).diff()
logR.head()
logR.plot()
logR.drop(logR.index[0], inplace = True)
logR.head()
numbins = int(1+ np.log2(logR.count())) 
logR.hist(bins = numbins, normed=True)

mu_log = np.mean(logR)
sigma_log = np.std(logR)
x_ticks = np.linspace(min(logR), max(logR),100)
plt.pyplot.plot(x_ticks, 1./((2.*np.pi)**0.5 *sigma_log)*np.exp(-((x_ticks - mu_log)/sigma_log)**2/2), 'r')
plt.pyplot.title('Histogram of log-retunrs and Gaussian with the same mean and variance')
