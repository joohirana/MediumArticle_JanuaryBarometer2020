
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import datetime
# quandl for financial data
import quandl
import pandas_datareader as dr
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[15]:


quandl.ApiConfig.api_key = 'your token'
sp = quandl.get("YALE/SPCOMP", authtoken="your token", collapse="daily")
sp.reset_index(inplace = True)


# In[16]:


sp.tail(12)


# In[134]:


sp['spreturn'] = sp['S&P Composite'].pct_change()


# In[135]:


sp.rename(columns={"Year": "date"}, inplace = True)


# In[136]:


sp['spreturn'].head()


# In[137]:


sp['sign'] = np.sign(sp['spreturn'])
#sp[sp['spreturn'] > 0] = 'pos'
#sp[sp['spreturn'] > 0] = 'neg'


# In[138]:


sp.head()


# In[139]:


totalyr = sp['date'].nunique()
totalyr


# In[140]:


sp.head()


# In[146]:


def successrate(df,m):
    df['date']= pd.to_datetime(df['date'],format='%m/%d/%Y')
    df['month']= pd.DatetimeIndex(df['date']).month
    df['year']= pd.DatetimeIndex(df['date']).year
    totalyr = df['year'].nunique()
    
    mon=[m,12.0]
    df_mon = df.groupby(df['month']).filter(lambda g: g['month'].isin(mon).all())
    
    sign = ['1']
    totalpos = df_mon.groupby('year').filter(lambda g: g['sign'].isin(sign).all())['year'].nunique()
    sucess_rate = totalpos/totalyr
    
    return sucess_rate,totalpos

jan_sucess_rate,jan_totalpos = successrate(sp,1)
jan_sucess_rate,jan_totalpos


# In[147]:


restofyr = sp[(sp['month'] == 1) | (sp['month'] == 12)]
restofyr['FebDecreturn'] = restofyr['S&P Composite'].pct_change()


# In[148]:


restofyr['FebDecreturn'] = restofyr['S&P Composite'].pct_change()


# In[149]:


restofyr


# In[150]:


restofyr_rtn = restofyr[(restofyr['month'] == 12)][['date','FebDecreturn','year']]
jan_sprtn = sp[sp['month'] == 1][['year','spreturn']]
sp2 = jan_sprtn.merge(restofyr_rtn)
sp2['jan_sign'] = np.sign(sp2['spreturn'])
sp2['FebDec_sign'] = np.sign(sp2['FebDecreturn'])


# In[151]:


jan_sprtn = sp[sp['month'] == 1][['year','spreturn']]


# In[152]:


sp2 = jan_sprtn.merge(restofyr_rtn)
sp2


# In[153]:


sp2['jan_sign'] = np.sign(sp2['spreturn'])
sp2['FebDec_sign'] = np.sign(sp2['FebDecreturn'])


# In[154]:


sp2


# sp2 is the main DF to use for SP500

# In[387]:


sp2[(sp2['jan_sign'] == 1)].year.count()
sp2[(sp2['jan_sign'] == 1) & (sp2['FebDec_sign'] == 1)].year.count()
sp2[(sp2['jan_sign'] == -1) & (sp2['FebDec_sign'] == -1)].year.count()
sp2[sp2['jan_sign'] == sp2['FebDec_sign']].year.count()


# In[389]:


sp2[(sp2['jan_sign'] == 1) & (sp2['FebDec_sign'] == 1)].year.count()


# In[390]:


sp2[(sp2['jan_sign'] == -1) & (sp2['FebDec_sign'] == -1)].year.count()


# In[327]:


sp_corr = sp2[sp2['jan_sign'] == sp2['FebDec_sign']].FebDecreturn.count()


# In[329]:


sp2.set_index('year',inplace = True)


# In[331]:


sp3 = sp2[sp2['jan_sign'] == sp2['FebDec_sign']][['spreturn','FebDecreturn']]
sp3


# In[472]:


sp3 = sp2[sp2['jan_sign'] == sp2['FebDec_sign']][['spreturn','FebDecreturn']]
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(40,20))
rects1 = ax.bar(sp3.index - width/2, sp3['spreturn'], width, label='Jan Return')
rects2 = ax.bar(sp3.index + width/2, sp3['FebDecreturn'], width, label='Feb-Dec Return')

ax.set_ylabel('Return', fontsize=40)
ax.set_xlabel('Year', fontsize=40)
ax.set_title("Jan Barometer on SP500",fontsize=40)
ax.legend(fontsize=40)
ax.tick_params(axis='both', which='major', labelsize=40)



# In[483]:


sp2[(sp2['jan_sign'] != sp2['FebDec_sign'])][['spreturn','FebDecreturn']].count()


# In[396]:


sp4 = sp2[(sp2['jan_sign'] != sp2['FebDec_sign']) & (sp2['jan_sign'] != 0)][['spreturn','FebDecreturn']]
sp4 = sp4.iloc[1:,:]


# In[408]:


#sp3 = sp2[sp2['jan_sign'] == sp2['FebDec_sign']][['spreturn','FebDecreturn']]
width = 0.95  # the width of the bars

fig, ax = plt.subplots(figsize=(40,20))
rects1 = ax.bar(sp4.index - width/2, sp4['spreturn'], width, label='Jan Return')
rects2 = ax.bar(sp4.index + width/2, sp4['FebDecreturn'], width, label='Feb-Dec Return')

ax.set_ylabel('Return',fontsize=40)
ax.set_xlabel('year',fontsize=40)
ax.set_title("Jan Barometer on SP500 - False Predictors",fontsize=40)
ax.legend(fontsize=40)
ax.tick_params(axis='both', which='major', labelsize=40)


# In[485]:


sp4['FebDecreturn'].nlargest(5)


# In[486]:


sp4['FebDecreturn'].nsmallest(5)


# In[329]:


totalyr = sp['year'].nunique()
totalyr


# Out of the 149 years, there were 143 year where jan had a positive return. but among that 63 years had positive december return

# In[327]:


sp[(sp['month'] == 1) & sp['sign'] == 1].count()


# # SP Total Returns

# In[276]:


first_sp = sp['S&P Composite'].iloc[11]
first_sp


# In[277]:


last_sp = sp['S&P Composite'].iloc[-1]
last_sp


# In[278]:


sp_tot_yrs= sp['year'].nunique()-1
sp_tot_yrs


# In[279]:


sp_tot_rtn = (last_sp-first_sp) / first_sp
sp_tot_rtn


# In[280]:


sp_fullyInvested = ((1+sp_tot_rtn)**(1/sp_tot_yrs)) - 1
sp_fullyInvested


# In[281]:


#25 postive years to invest
sp_pos = sp2[(sp2['jan_sign'] == 1)]
sp_pos


# In[282]:


sp_posYr = sp2[(sp2['jan_sign'] == 1)].FebDecreturn.count()
sp_posYr


# In[283]:


sp_pos['rnt1']= sp_pos['FebDecreturn']+1
sp_sum = sp_pos['rnt1'].sum()
sp_sum
sp_gm = (sp_sum ** (1/sp_posYr) ) - 1
sp_gm


# In[284]:


sp_gm = (sp_sum ** (1/sp_posYr) ) - 1
sp_gm


# In[285]:


sp_fullyInvested


# # Jan Barometer on different indexes

# NASDAQ Composite:

# In[286]:


df = dr.data.get_data_yahoo('^IXIC', start='1971-12-31', end = '2019-12-31')
df.reset_index(inplace = True)
df['month'] = pd.DatetimeIndex(df['Date']).month
df['year'] = pd.DatetimeIndex(df['Date']).year


# In[287]:


df


# In[288]:


df =df.groupby(['year', 'month']).last().reset_index()
df['nasreturn'] = df['Close'].pct_change()


# In[289]:


tot_yrs= df['year'].nunique()-1
tot_yrs


# In[290]:


first = df['Close'][0]
first


# In[291]:


last = df['Close'].iloc[-1]
last


# In[292]:


tot_rtn = (last-first) / first
tot_rtn


# In[293]:


nas_fullyInvested = ((1+tot_rtn)**(1/tot_yrs)) - 1
nas_fullyInvested


# In[336]:


nas_corr = nas2[(nas2['jan_sign'] == nas2['FebDec_sign'])].FebDecreturn.count()


# In[337]:


nas2[(nas2['jan_sign'] == 1)].count()


# In[338]:


#31 postive years to invest
a = nas2[(nas2['jan_sign'] == 1)]
a


# In[339]:


a['rnt1']= a['FebDecreturn']+1


# In[340]:


sum1 = a['rnt1'].sum()


# In[341]:


nas_gm = (sum1 ** (1/31) ) - 1
nas_gm


# In[342]:


nas_fullyInvested


# In[335]:


nas_restofyr = df[(df['month'] == 1) | (df['month'] == 12)]
nas_restofyr['FebDecreturn'] = nas_restofyr['Close'].pct_change()

restofyr_rtn = nas_restofyr[(nas_restofyr['month'] == 12)][['Date','FebDecreturn','year']]
jan_nasrtn = df[df['month'] == 1][['year','nasreturn']]
nas2 = jan_nasrtn.merge(restofyr_rtn)
nas2['jan_sign'] = np.sign(nas2['nasreturn'])
nas2['FebDec_sign'] = np.sign(nas2['FebDecreturn'])


# In[503]:


nas2


# In[505]:


nas2[(nas2['jan_sign'] == 1)].year.count()


# In[506]:


nas2[(nas2['jan_sign'] == 1) & (nas2['FebDec_sign'] == 1)].year.count()


# In[507]:


nas2[(nas2['jan_sign'] == -1) & (nas2['FebDec_sign'] == -1)].year.count()


# In[510]:


worked = nas2[nas2['jan_sign'] == nas2['FebDec_sign']].year.count()
worked


# In[556]:


nasYr2Invest = nas2[(nas2['jan_sign'] == 1) & (nas2['FebDec_sign'] == 1)].index.to_list()


# In[511]:


total = nas2['year'].count()
worked/total


# In[514]:


nas2.set_index('year', inplace=True)


# In[521]:


nas3 = nas2[nas2['jan_sign'] == nas2['FebDec_sign']][['nasreturn','FebDecreturn']]
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(20,10))
rects1 = ax.bar(nas3.index - width/2, nas3['nasreturn'], width, label='Jan Return')
rects2 = ax.bar(nas3.index + width/2, nas3['FebDecreturn'], width, label='Feb-Dec Return')

ax.set_ylabel('Return')
ax.set_title("Jan Barometer on SP500 - Correct Predictions")
ax.legend()


# In[522]:


nas4 = nas2[nas2['jan_sign'] != nas2['FebDec_sign']][['nasreturn','FebDecreturn']]
width = 0.75  # the width of the bars

fig, ax = plt.subplots(figsize=(20,10))
rects1 = ax.bar(nas4.index - width/2, nas4['nasreturn'], width, label='Jan Return')
rects2 = ax.bar(nas4.index + width/2, nas4['FebDecreturn'], width, label='Feb-Dec Return')

ax.set_ylabel('Return')
ax.set_title("Jan Barometer on SP500 - False Predictions")
ax.legend()


# # Dow Jones Industry

# In[298]:


dji = dr.data.get_data_yahoo('^DJI', start='1984-12-31', end = '2019-12-31')


# In[299]:


dji.reset_index(inplace = True)
dji['month'] = pd.DatetimeIndex(dji['Date']).month
dji['year'] = pd.DatetimeIndex(dji['Date']).year
df_dji = dji.groupby(['year', 'month']).last().reset_index()
df_dji['return'] = df_dji['Close'].pct_change()
df_dji['sign'] = np.sign(df_dji['return'])


# In[300]:


df_dji


# In[301]:


dow_restofyr = df_dji[(df_dji['month'] == 1) | (df_dji['month'] == 12)]
dow_restofyr['FebDecreturn'] = dow_restofyr['Close'].pct_change()

dow_restofyr_rtn = dow_restofyr[(dow_restofyr['month'] == 12)][['Date','FebDecreturn','year']]
jan_dowrtn = df_dji[df_dji['month'] == 1][['year','return']]
dow2 = jan_dowrtn.merge(dow_restofyr_rtn)
dow2['jan_sign'] = np.sign(dow2['return'])
dow2['FebDec_sign'] = np.sign(dow2['FebDecreturn'])


# In[302]:


dow2


# In[303]:


dow_tot_yrs= dow2['year'].nunique()-1
dow_tot_yrs


# In[304]:


dji


# In[305]:


dow_first = dji['Close'].iloc[0]
dow_last = dji['Close'].iloc[-1]
dow_last

dow_tot_rtn = (dow_last-dow_first) / dow_first
dow_tot_rtn


# In[306]:


dow_tot_yr = dji['year'].nunique() -1
dow_tot_yr


# In[307]:


dow_fullyInvested = ((1+dow_tot_rtn)**(1/dow_tot_yr)) - 1
dow_fullyInvested


# In[308]:


dow_corr = dow2[(dow2['jan_sign'] ==dow2['FebDec_sign'])].year.count()


# In[309]:


dow_posyr = dow2[(dow2['jan_sign'] == 1) ].year.count()
dow_posyr


# In[310]:


dow_pos = dow2[(dow2['jan_sign'] == 1)]
dow_pos


# In[311]:


dow_pos['rnt1']= dow_pos['FebDecreturn']+1
dow_pos


# In[312]:


dow_sum = dow_pos['rnt1'].sum()

dow_gm = (dow_sum ** (1/dow_posyr) ) - 1
dow_gm


# # Russel 2000

# In[315]:


rut = dr.data.get_data_yahoo('^RUT', start='1987-12-31', end = '2019-12-31')
rut


# In[316]:


rut.reset_index(inplace = True)
rut['month'] = pd.DatetimeIndex(rut['Date']).month
rut['year'] = pd.DatetimeIndex(rut['Date']).year
df_rut = rut.groupby(['year', 'month']).last().reset_index()
df_rut['return'] = df_rut['Close'].pct_change()
df_rut['sign'] = np.sign(df_rut['return'])


# In[317]:


df_rut


# In[318]:


rut_restofyr = df_rut[(df_rut['month'] == 1) | (df_rut['month'] == 12)]
rut_restofyr['FebDecreturn'] = rut_restofyr['Close'].pct_change()

rut_restofyr_rtn = rut_restofyr[(rut_restofyr['month'] == 12)][['Date','FebDecreturn','year']]
jan_rutrtn = df_rut[df_rut['month'] == 1][['year','return']]
rut2 = jan_rutrtn.merge(rut_restofyr_rtn)
rut2['jan_sign'] = np.sign(rut2['return'])
rut2['FebDec_sign'] = np.sign(rut2['FebDecreturn'])


# In[347]:


rut_first = rut['Close'].iloc[0]
rut_last = rut['Close'].iloc[-1]
rut_tot_rnt = ((rut_last-rut_first)/ rut_first) 
rut_tot_rnt
rut_fullyInvested = ((1+rut_tot_rnt)**(1/rut_tot_yr)) - 1
rut_fullyInvested


# In[348]:


rut_tot_yr = df_rut.year.nunique() #33


# In[349]:


rut_corr = rut2[(rut2['jan_sign'] == rut2['FebDec_sign'])].year.count()


# In[350]:


rut_corr/rut_tot_yr


# In[351]:


rut_posyr = rut2[(rut2['jan_sign'] == 1)].year.count()
rut_posyr
rut_pos = rut2[(rut2['jan_sign'] == 1)]
rut_pos
rut_pos['rnt1'] = rut_pos['FebDecreturn'] +1 
rut_sum = rut_pos['rnt1'].sum()
rut_gm = (rut_sum ** (1/rut_posyr) ) - 1


# In[352]:


rut_sum = rut_pos['rnt1'].sum()
rut_sum


# In[353]:


rut_gm = (rut_sum ** (1/rut_posyr) ) - 1
rut_gm


# # Index Summary Chart

# In[354]:


index = pd.DataFrame(data = {'index': ['SP500','NASDAQ', 'DOW JONES', 'RUSSEL 2000'],
                            'TotalYears':[sp_tot_yrs,tot_yrs,dow_tot_yr,rut_tot_yr],
                            'Correctly Predicted Years': [sp_corr,nas_corr,dow_corr,rut_corr],
                             'Total_Return': [sp_fullyInvested,nas_fullyInvested,dow_fullyInvested,rut_fullyInvested],
                             'Barometer_Return': [sp_gm,nas_gm,dow_gm,rut_gm]
                            
                            })


# In[359]:


index['Success_Rate'] = index['Correctly Predicted Years']/index['TotalYears']


# In[361]:


index.style.format({'Total_Return': "{:.2%}",'Barometer_Return': "{:.2%}",'Success_Rate': "{:.2%}"})


# In[417]:


index.set_index('index')[['Total_Return','Barometer_Return']].plot(figsize=(20, 8), linewidth=2.5,marker='o', markerfacecolor='black')
plt.ylabel("Return",fontsize=20)
plt.xlabel("index",fontsize=20)
plt.title("Following the Barometer or Ignoring It - Annualized Returns by Index", y=1.02, fontsize=30);
#df.plot(style='.-')

#plt.xlabel(fontsize=40)

plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)


# # Sector SP500 breakdown

# In[35]:


tech = dr.data.get_data_yahoo('XLK', start='1998-12-31', end = '2019-12-31')
consumer_dis = dr.data.get_data_yahoo('XLY', start='1998-12-31', end = '2019-12-31')
healthcare = dr.data.get_data_yahoo('XLV', start='1998-12-31', end = '2019-12-31')
utilities = dr.data.get_data_yahoo('XLU', start='1998-12-31', end = '2019-12-31')
consumer_staples = dr.data.get_data_yahoo('XLP', start='1998-12-31', end = '2019-12-31')
#reits = dr.data.get_data_yahoo('XLRE', start='1998-12-31', end = '2019-12-31') #only till 2015
industrials = dr.data.get_data_yahoo('XLI', start='1998-12-31', end = '2019-12-31')
financials = dr.data.get_data_yahoo('XLF', start='1998-12-31', end = '2019-12-31')
#media = dr.data.get_data_yahoo('XLC', start='1998-12-31', end = '2019-12-31') #only till 2018
materials = dr.data.get_data_yahoo('XLB', start='1998-12-31', end = '2019-12-31')
energy = dr.data.get_data_yahoo('XLE', start='1998-12-31', end = '2019-12-31')


# In[18]:


tech


# In[36]:


def sector_rate(df):
    df.reset_index(inplace = True)
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['year'] = pd.DatetimeIndex(df['Date']).year
    df_2 = df.groupby(['year', 'month']).last().reset_index()
    df_2['return'] = df_2['Close'].pct_change()
    df_2['sign'] = np.sign(df_2['return'])
    
    df_restofyr = df_2[(df_2['month'] == 1) | (df_2['month'] == 12)]
    df_restofyr['FebDecreturn'] = df_restofyr['Close'].pct_change()

    df_restofyr_rtn = df_restofyr[(df_restofyr['month'] == 12)][['Date','FebDecreturn','year']]
    jan_dfrtn = df_2[df_2['month'] == 1][['year','return']]
    df2 = jan_dfrtn.merge(df_restofyr_rtn)
    df2['jan_sign'] = np.sign(df2['return'])
    df2['FebDec_sign'] = np.sign(df2['FebDecreturn'])
    
    
    totalyr = df_2['year'].nunique() -1
    
    df_corr = df2[(df2['jan_sign'] == df2['FebDec_sign'])].year.count()
    df_tp = df2[(df2['jan_sign'] == 1) & (df2['FebDec_sign'] == 1)].year.count()
    df_fp = df2[(df2['jan_sign'] == 1) & (df2['FebDec_sign'] == -1)].year.count()
    df_fn = df2[(df2['jan_sign'] == -1) & (df2['FebDec_sign'] == 1)].year.count()
    df_tn = df2[(df2['jan_sign'] == -1) & (df2['FebDec_sign'] == -1)].year.count()
    
    sucess_rate = df_corr/totalyr
    
    df_first = df['Close'].iloc[0]
    df_last = df['Close'].iloc[-1]
    df_tot_rnt = ((df_last-df_first)/ df_first) -1
    #df_tot_rnt
    df_fullyInvested = ((1+df_tot_rnt)**(1/totalyr)) - 1
    
    #df_fullyInvested
    
    df_posyr = df2[(df2['jan_sign'] == 1)].year.count()
 
    df_pos = df2[(df2['jan_sign'] == 1)]
    df_pos['rnt1'] = df_pos['FebDecreturn'] +1 
    df_sum = df_pos['rnt1'].sum()
    #df_gm = (df_sum ** (1/df_posyr) ) - 1
    df_gm = df_pos['FebDecreturn'].mean()
    
    return sucess_rate,df_corr,df_fullyInvested,df_gm,df_tp,df_fp,df_fn,df_tn


# In[37]:


sector = [tech,consumer_dis,healthcare, utilities, consumer_staples,industrials, financials,materials,energy]


# In[38]:


df_sector = pd.DataFrame({'Sector': ['tech','consumer_dis','healthcare', 'utilities', 'consumer_staples','industrials', 'financials','materials','energy']      
                            })


# In[39]:


df_sector


# In[40]:


sucess_rate_lst = []
df_corr_lst = []
df_fullyInvested_lst = []
df_gm_lst = []
df_tp_lst = []
df_fp_lst = []
df_fn_lst = []
df_tn_lst = []
for s in sector:
    #print(s)
    sucess_rate,df_corr,df_fullyInvested,df_gm,df_tp,df_fp,df_fn,df_tn = sector_rate(s)
    #print(sucess_rate)
    sucess_rate_lst.append(sucess_rate)
    df_corr_lst.append(df_corr)
    df_fullyInvested_lst.append(df_fullyInvested)
    df_gm_lst.append(df_gm)
    df_tp_lst.append(df_tp)
    df_fp_lst.append(df_fp)
    df_fn_lst.append(df_fn)
    df_tn_lst.append(df_tn)
    
    
    
    #df_sector2['sucess_rate'] = df_sector2['sucess_rate'].append(sucess_rate)
    
    #data = [s,sucess_rate,df_corr,df_fullyInvested,df_gm,df_tp,df_fp,df_fn,df_tn]
    #df_sector = df_sector.append(data)
    #df_sector['sucess_rate'] =sucess_rate
    
    


# In[453]:


df_sector['success_rate'] = sucess_rate_lst
df_sector['Correctly Predicted Years'] = df_corr_lst 
df_sector['Ignore Jan Bar Return'] = df_fullyInvested_lst 
df_sector['Jan Bar Return'] = df_gm_lst 
df_sector['True Positives'] = df_tp_lst 
df_sector['False Positives'] = df_fp_lst 
df_sector['False Negatives'] = df_fn_lst 
df_sector['True Negatives'] = df_tn_lst 


# In[454]:


df_sector


# In[467]:


df_sector[['Sector','success_rate','Correctly Predicted Years']].style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'success_rate': "{:.2%}"})


# In[272]:


df_sector[['Sector','Ignore Jan Bar Return','Jan Bar Return']].style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39').format({'Ignore Jan Bar Return': "{:.2%}",'Jan Bar Return': "{:.2%}"})


# In[54]:


df_sector


# In[55]:


df_sector.set_index('Sector')[['Ignore Jan Bar Return','Jan Bar Return']].plot(figsize=(12, 5), linewidth=2.5,marker='o', markerfacecolor='black')
plt.ylabel("Return")
plt.title("Following the Barometer or Ignoring It - Annualized Returns by Sector", y=1.02, fontsize=22);
#df.plot(style='.-')


# In[430]:



# create plot
n_groups = 9
fig, ax = plt.subplots(figsize=(20,10))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, df_sector['Jan Bar Return'], bar_width,
alpha=opacity,
color='m',
label='Barometer Return')

rects2 = plt.bar(index + bar_width, df_sector['Ignore Jan Bar Return'], bar_width,
alpha=opacity,
color='y',
label='Ignore Barometer Return')


plt.ylabel('Returns', fontsize=16)
plt.title('Following the Barometer or Ignoring It - Annualized Returns by Sector',fontsize=18)
plt.xticks(index + bar_width, df_sector['Sector'], size=14)
plt.legend(fontsize=18)

def autolabel(rects):
    for rect in rects:
        height = np.round(rect.get_height(),decimals=3)
        ax.annotate('{:.2%}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)


autolabel(rects1)
autolabel(rects2)
plt.tick_params(axis='both', which='major', labelsize=15)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])


# In[57]:


df_sector


# In[86]:


fig, ax1 = plt.subplots(figsize=(15, 5))

df_sector.set_index('Sector')['Correctly Predicted Years'].plot(kind='bar', color='y')
df_sector.set_index('Sector')['sucess_rate'].plot(kind='line', marker='d', secondary_y=True)
ax1.set_ylabel("Correctly Predicted Years")
ax2.set_ylabel("Success rate")


# In[439]:


fig, ax1 = plt.subplots(figsize=(15, 5))
plt.style.use('seaborn-white')
ax2 = ax1.twinx()
chart = df_sector.set_index('Sector')['Correctly Predicted Years'].plot(kind='bar',color = 'y', ax=ax1)
df_sector.set_index('Sector')['sucess_rate'].plot(kind='line', marker='d', ax=ax2)

ax1.set_ylabel("Correctly Predicted # of Years",fontsize=14)
ax2.set_ylabel("Success rate",fontsize=14)
ax1.set_ylim([0,16])
chart.set_xticklabels(chart.get_xticklabels(), rotation=45,fontsize=16)
ax2.legend(loc=9, bbox_to_anchor=(1, -0.2),fontsize=14)
ax1.legend(loc=9, bbox_to_anchor=(1, -0.3),fontsize=14)
plt.title("Jan Barometer Success Rate by Sectors", y=1.02, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=12)
ax1.tick_params(axis='both', which='major', labelsize=12)


# In[131]:


error = df_sector[['Sector', 'True Positives', 'False Positives','False Negatives','True Negatives']]


# In[132]:


error.set_index('Sector', inplace=True)


# In[446]:


#fig, ax1 = plt.subplots(figsize=(15, 5))
chart2 = error.plot.bar(stacked=True,figsize=(15, 5))
chart2.set_xticklabels(chart.get_xticklabels(), rotation=45,size=14)
chart2.legend(loc=9, bbox_to_anchor=(1.1, 1),fontsize=14)

plt.title("Confusion Matrix by Sector", y=1.02, fontsize=22);

for p in chart2.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    
    chart2.text(x+width/2, 
            y+height/2, 
            height, 
            horizontalalignment='center', 
            verticalalignment='center', fontsize=13)
chart2.set_ylabel("# of Years", fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)


# In[128]:


error['TP + TN'] = error['True Positives'] + error['True Negatives'] 
error['FP + FN'] = error['False Positives'] + error['False Negatives']


# In[129]:


error[[ 'TP + TN', 'FP + FN']]


# # Jan Bar on Past Epidemics

# In[155]:


sp2


# In[252]:


epidemics = [1889,1916,1918,1957,1981,2009,2014,2015]


# In[253]:


epidemic_df = sp2[sp2['year'].isin(epidemics)]
epidemic_df


# In[254]:


epidemic_df.rename(columns={"spreturn": "JanReturn", 'FebDecreturn': 'FebDecReturn'}, inplace=True)


# In[255]:


epidemic_df.round({'JanReturn': 3, 'FebDecReturn': 3})


# In[256]:


epidemic_df.style.format({'JanReturn': "{:.2%}",'FebDecReturn': "{:.2%}"})


# In[257]:


epidemic_df['Epidemic/Pandemic'] = ['Influenza Pandemic','American Polio Epidemic','Spanish Flu','Asian Flu','AIDS pandemic','H1N1 Swine Flu','Ebola Epidemic','Zika Virus epidemic']


# In[258]:


epidemic_df


# In[259]:


epidemic_df[epidemic_df['jan_sign'] == epidemic_df['FebDec_sign']].year.count()


# In[260]:


5/8


# In[261]:


epidemic_df[['Epidemic/Pandemic', 'year','JanReturn','FebDecReturn']].style.format({'JanReturn': "{:.2%}",'FebDecReturn': "{:.2%}"})


# In[262]:


chart_df= epidemic_df[['Epidemic/Pandemic','JanReturn','FebDecReturn']].set_index('Epidemic/Pandemic')
chart_df


# In[452]:


fig, ax = plt.subplots(figsize=(25,10))
chart_df.plot.bar(rot=0,ax = ax,fontsize =16)
ax.set_ylabel('Return',fontsize =16)
ax.set_xlabel('Epidemics/Pandemics',fontsize =16)
plt.style.use('seaborn')
ax.set_title("Jan Barometer on Past Epidemic/Pandemic", fontsize =22)

ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=18)


# In[201]:


chart_df.index


# In[222]:


sp2


# In[223]:


jan2020 = dr.data.get_data_yahoo('^GSPC', start='2020-01-01', end = '2020-1-31')


# In[226]:


jan2020


# In[473]:


#used wsj 2020 numbers 
first = jan2020.Close.iloc[0]
first = 3257.85


# In[474]:


#used wsj 2020 numbers 
last = jan2020.Close.iloc[-1]
last = 3225.52


# In[475]:


janrnt = (last-first)/first
janrnt


# In[476]:


epidemic_df2 = epidemic_df.append({'year':2020,'Epidemic/Pandemic' : 'COVID-19 pandemic' , 'JanReturn' : janrnt,'FebDecReturn': 0} , ignore_index=True)


# In[477]:


epidemic_df2


# In[478]:


epidemic_df[['Epidemic/Pandemic', 'year','JanReturn','FebDecReturn']].style.format({'JanReturn': "{:.2%}",'FebDecReturn': "{:.2%}"})


# In[479]:


a = epidemic_df2[['Epidemic/Pandemic', 'year','JanReturn','FebDecReturn']].style.format({'JanReturn': "{:.2%}",'FebDecReturn': "{:.2%}"})


# In[480]:


a

