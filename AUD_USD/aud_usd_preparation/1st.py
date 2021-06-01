import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime,timedelta
import datetime

#IMPORT ALL THE FILES AND KEEP ONLY DESIRERED COLUMNS OF THE EVENT DATA

pd.options.mode.chained_assignment = None  #  no important. This is for avoiding the warning of chained assignments

drop_cols = ['c','D','E','F','G','I','J']

AUD_USD_df = pd.read_csv('AUDUSD-2000-2020-15m.csv') #AUS_USD CURRENCY DATA SET

# import all the event files, variable name tells what is the event
#use 'parse_dates' keyword ynside the 'read_csv' function in order to combine date and time also convert the data type into 'datetime64'
AUD_GDP_df = pd.read_csv('AUD_GDP.csv',header = None,names =['DATE','TIME','c','D','E','F','G','AUD_GDP','I','J'] ,parse_dates= [[0,1]])
AUD_GDP_df = AUD_GDP_df.drop(columns=drop_cols)

AUD_PPI_df = pd.read_csv('AUD_PPI.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','AUD_PPI','I','J'],parse_dates= [[0,1]])
AUD_PPI_df = AUD_PPI_df.drop(columns=drop_cols)

AUD_RETAILSALES_df = pd.read_csv('AUD_RETAILSALES.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','AUD_RETAILSALES','I','J'],parse_dates= [[0,1]])
AUD_RETAILSALES_df = AUD_RETAILSALES_df.drop(columns=drop_cols)

AUD_UNEMP_df = pd.read_csv('AUD_UNEMP.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','AUD_UNEMP','I','J'],parse_dates= [[0,1]])
AUD_UNEMP_df = AUD_UNEMP_df.drop(columns=drop_cols)

AUD_CPI_df = pd.read_csv('AUD_CPI.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','AUD_CPI','I','J'],parse_dates= [[0,1]])
AUD_CPI_df = AUD_CPI_df.drop(columns=drop_cols)

USD_CPI_df = pd.read_csv('USD_CPI.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_CPI','I','J'],parse_dates= [[0,1]])
USD_CPI_df = USD_CPI_df.drop(columns=drop_cols)

USD_GDP_df = pd.read_csv('USD_GDP.csv',header = None,names =['DATE','TIME','c','D','E','F','G','USD_GDP','I','J'],parse_dates= [[0,1]])
USD_GDP_df = USD_GDP_df.drop(columns=drop_cols)

USD_IR_df = pd.read_csv('USD_IR.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_IR','I','J'],parse_dates= [[0,1]])
USD_IR_df = USD_IR_df.drop(columns=drop_cols)

USD_PAYROLL_df = pd.read_csv('USD_PAYROLL.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_PAYROLL','I','J'],parse_dates= [[0,1]])
USD_PAYROLL_df = USD_PAYROLL_df.drop(columns=drop_cols)

USD_PPI_df = pd.read_csv('USD_PPI.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_PPI','I','J'],parse_dates= [[0,1]])
USD_PPI_df = USD_PPI_df.drop(columns=drop_cols)

USD_RETAIL_df = pd.read_csv('USD_RETAIL.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_RETAIL','I','J'],parse_dates= [[0,1]])
USD_RETAIL_df = USD_RETAIL_df.drop(columns=drop_cols)

USD_UNEMP_df = pd.read_csv('USD_UNEMP.csv',header = None,names = ['DATE','TIME','c','D','E','F','G','USD_UNEMP','I','J'],parse_dates= [[0,1]])
USD_UNEMP_df = USD_UNEMP_df.drop(columns=drop_cols)

#select the data from 2011-2020 from the 'AUD_USD_df' data set where the pips are located.

d =AUD_USD_df.loc[272142:500027 ,:]
df = d.reset_index(drop=True)
df['EX_DATE_TIME'] = pd.to_datetime(df['DATE_TIME']) # convert the type to datetime
df['DATE_TIME'] = df['EX_DATE_TIME']- pd.Timedelta(hours=7)
 # substract 7 hours from the currency set

#merge all the dataframes (all the events of both countries and market prices by matchin

new_df =  df.merge(AUD_GDP_df, on = 'DATE_TIME', how = 'left')\
          .merge(AUD_PPI_df, on = 'DATE_TIME', how = 'left')\
            .merge(AUD_RETAILSALES_df, on = 'DATE_TIME', how = 'left')\
            .merge(AUD_UNEMP_df, on = 'DATE_TIME', how = 'left')\
            .merge(AUD_CPI_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_PPI_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_CPI_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_GDP_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_IR_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_PAYROLL_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_RETAIL_df, on = 'DATE_TIME', how = 'left')\
            .merge(USD_UNEMP_df, on = 'DATE_TIME', how = 'left')

#checking whether is there any overlapping events in the merged dataframe


temp_df = new_df.drop(columns = ['DATE_TIME','HIGH','LOW','OPEN','CLOSE','EX_DATE_TIME']).notnull().sum(axis=1) # A Series of counting number of non 'nan' values accross a row

#A function to checking number of overlap events

def OverlapCounter(t_df):
    count = 0
    for i in t_df:
        if i>1:
            count+=1
    print('There are ', count, ' overlapping events')


OverlapCounter(temp_df)# count overlap events

t_df = new_df[temp_df==1] #avoid overlapping non event columns
df_with_reset_index = t_df.reset_index(drop=True)
df_with_reset_index
x= df_with_reset_index.drop(columns = ['DATE_TIME','HIGH','LOW','OPEN','CLOSE','EX_DATE_TIME'])

event_type = []
event_values=[]
for i in range(len(x)):
    for col in x.columns:
        val = x.loc[i,col]
        if str(val) != 'nan':
            event_type.append(col)

            event_values.append(val)

df_with_reset_index['Event_value'] = event_values
df_with_reset_index['Event_type'] = event_type

# rearrange the data frame by keeping only desired columns
train_df = df_with_reset_index[['DATE_TIME','HIGH', 'LOW','OPEN','CLOSE', 'Event_value','Event_type']]

for i in range(8):

    tempor_df = pd.DataFrame({}) # created a temporary dataframe in order to store shifting dates

    tempor_df['DATE_TIME'] = train_df['DATE_TIME'] + pd.Timedelta(minutes=15*(i+1)) #shift the time by 15 minutes of slicers

    x = pd.merge(df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE']],tempor_df, how='right', on='DATE_TIME') # merge the shifted date column and join HIGH, LOW values in the pip data set

    x_1 = x['HIGH'].fillna(method = 'bfill') # filling 'nan' values with the next value respected to the 'nan'
    x_2 = x['LOW'].fillna(method= 'bfill')
    x_3 = x['CLOSE'].fillna(method = 'bfill')
    x_4 = x['OPEN'].fillna(method = 'bfill')

    col_name = 'after_'+ str(15*(i+1)) + '_mins'#column names that included (open+close)/2 values 0-120 min in time duration

    train_df[col_name] = (x_1 + x_2 + x_3 + x_4)/4

my_df = pd.DataFrame({})

init_value = (train_df.iloc[:,1] + train_df.iloc[:,2] + train_df.iloc[:,3] +train_df.iloc[:,4])/4

for i in range(8):
    col = 'defference of avg from_' + str((i)*15) +'_to_' + str((i+1)*15)


    my_df[col] = abs(train_df.iloc[:,i+7] - init_value)

    init_value = train_df.iloc[:,i+7]

train_df = train_df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE','Event_value','Event_type']]

col = my_df.columns
new_s = my_df[col].apply(lambda row: ' '.join(row.values.astype(str)), axis=1) # combine all the values in a row in the train_df into a one string

def timeDuration(x):
    a = x.split()
    l = list(map(float,a))
    y = ((l[0]+l[1])/2)*0.75
    l.pop(0)
    l.pop(0)
    l = [x for x in l if x>=y ]
    if len(l) >=3:
        p = 'Long term'
    else:
        p= 'Short term'

    return p

train_df['Time duration'] = new_s.apply(lambda x: timeDuration(x))

   #find the trend when an event happen
def trend(x):
    if x == datetime.datetime(2020,3,18,20,30,0):
        final = 0
    else:
        initial_value  =  x - datetime.timedelta(hours= 25)


        init_check = df[df['DATE_TIME'] == initial_value]['OPEN']


        forward_factor = 0
        backward_factor = 0

        while len(init_check) != 1:
            delta_time = datetime.timedelta(minutes=15)
            initial_value = initial_value + delta_time
            init_check = df[df['DATE_TIME'] == initial_value]['OPEN']
            forward_factor+= 1


        initial_value  =  x - datetime.timedelta(hours= 25)
        init_check = df[df['DATE_TIME'] == initial_value]['OPEN']
        while len(init_check) !=1:
            delta_time = datetime.timedelta(minutes=15)
            initial_value = initial_value - delta_time
            init_check = df[df['DATE_TIME'] == initial_value]['OPEN']
            backward_factor +=1


        net_factor = backward_factor + forward_factor
        initial_value = initial_value - datetime.timedelta(minutes=15)*(net_factor)

        max_= 0
        min_ = 1000
        for i in range(99):
            time_delta = datetime.timedelta(minutes=15*(99-i))
            date = x - time_delta
            s = df[df['DATE_TIME']== date]['OPEN']

            if len(s)==1:
                date = date

            else:
                time_shift = datetime.timedelta(minutes=15)* (net_factor)
                date = date - time_shift


            avg_1 =(df[df['DATE_TIME'] == date ]['OPEN'] + df[df['DATE_TIME']==date ]['CLOSE'])/2
            avg_2 = (df[df['DATE_TIME'] == initial_value ]['OPEN'] + df[df['DATE_TIME']== initial_value ]['CLOSE'])/2



            if len(avg_1) !=1  or len(avg_2) !=1:
                val = val

            else:
                val = abs(float(avg_1) - float(avg_2))




            if val> max_:
                max_=val
            else:
                max_ = max_

            if val < min_:
                min_ = val
            else:
                min_ = min_

            initial_value = date





        initial_value  =  x + datetime.timedelta(hours= 25)


        init_check = df[df['DATE_TIME'] == initial_value]['OPEN']

        forward_factor = 0
        backward_factor = 0

        while len(init_check) != 1:
            delta_time = datetime.timedelta(minutes=15)
            initial_value = initial_value + delta_time
            init_check = df[df['DATE_TIME'] == initial_value]['OPEN']
            forward_factor+= 1



        initial_value  =  x + datetime.timedelta(hours= 25)
        init_check = df[df['DATE_TIME'] == initial_value]['OPEN']

        while len(init_check) !=1:
            delta_time = datetime.timedelta(minutes=15)
            initial_value = initial_value - delta_time
            init_check = df[df['DATE_TIME'] == initial_value]['OPEN']
            backward_factor +=1



        net_factor = backward_factor + forward_factor
        initial_value  =  x + datetime.timedelta(hours= 25)
        initial_value = initial_value + datetime.timedelta(minutes=15)*(net_factor)

        for i in range(100):
            time_delta = datetime.timedelta(minutes=15*(99-i))
            date = x + time_delta
            s = df[df['DATE_TIME']== date]['OPEN']

            if len(s)==1:
                date = date

            else:
                time_shift = datetime.timedelta(minutes=15)* (net_factor)
                date = date + time_shift


            avg_1 =(df[df['DATE_TIME'] == date ]['OPEN'] + df[df['DATE_TIME']==date ]['CLOSE'])/2
            avg_2 = (df[df['DATE_TIME'] == initial_value ]['OPEN'] + df[df['DATE_TIME']== initial_value ]['CLOSE'])/2

            if len(avg_1) !=1  or len(avg_2) !=1 :
                val = val
            else:
                 val = abs(float(avg_1) - float(avg_2))

            if val> max_:
                max_=val
            else:
                max_ = max_

            if val < min_:
                min_ = val
            else:
                min_ = min_

            initial_value = date
        final = min_ + (max_ - min_)*0.10

    return final


train_df['Boundaries'] = train_df['DATE_TIME'].apply(lambda x : trend(x))
for i in range(2):

    tempor_df = pd.DataFrame({}) # created a temporary dataframe in order to store shifting dates

    tempor_df['DATE_TIME'] = train_df['DATE_TIME'] - pd.Timedelta(minutes=15*(2-i)) #shift the time by 15 minutes of slicers

    x = pd.merge(df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE']],tempor_df, how='right', on='DATE_TIME') # merge the shifted date column and join HIGH, LOW values in the pip data set


    x_1 = x['CLOSE'].fillna(method = 'bfill')
    x_2 = x['OPEN'].fillna(method = 'bfill')

    col_name = '_before'+ str(15*(2-i)) + '_mins' + '_in_trend'#column names that included (open+close)/2 values 0-120 min in time duration

    train_df[col_name] = (x_1 + x_2 )/2

for i in range(8):

    tempor_df = pd.DataFrame({}) # created a temporary dataframe in order to store shifting dates

    tempor_df['DATE_TIME'] = train_df['DATE_TIME'] + pd.Timedelta(minutes=15*(1+i)) #shift the time by 15 minutes of slicers

    x = pd.merge(df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE']],tempor_df, how='right', on='DATE_TIME') # merge the shifted date column and join HIGH, LOW values in the pip data set


    x_1 = x['CLOSE'].fillna(method = 'bfill')
    x_2 = x['OPEN'].fillna(method = 'bfill')

    col_name = '_after'+ str(15*(1+i)) + '_mins' + '_in_trend'#column names that included (open+close)/2 values 0-120 min in time duration

    train_df[col_name] = (x_1 + x_2 )/2

train_df
my_df = pd.DataFrame({})

init_value = (train_df.iloc[:,3] +train_df.iloc[:,4])/2

for i in range(8):
    col = 'defference of avg in ' +  'slot ' + str(i+1) + ' in trend'


    my_df[col] = train_df.iloc[:,i+9] - init_value

    init_value = train_df.iloc[:,i+9]
my_df
trend = []

def final_trend(x):
    return {up_count: 'UP' ,down_count: 'DOWN', range_count: 'RANGE'}.get(x)

for i in range(len(my_df)):
    up_count = 0
    down_count = 0
    range_count = 0
    for col in my_df.columns :
        val = my_df.loc[i,col]
        lower_boundary = (-1)*(train_df['Boundaries'][i])
        upper_boundary = train_df['Boundaries'][i]
        if val > upper_boundary:
            up_count+=1
        elif val< lower_boundary :
            down_count+=1
        else:
            range_count+=1
    x = max(up_count,down_count,range_count)
    trend.append(final_trend(x))

train_df['Trend'] = trend

train_df= train_df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE','Event_value','Event_type','Time duration','Trend','Boundaries']]

x= train_df[train_df['Trend']== 'RANGE'].reset_index(drop=True)
x['val']=0

for i in range(11):
    t= pd.DataFrame({})
    t['DATE_TIME']= x['DATE_TIME']+pd.Timedelta(minutes = (-45) + (i+1)*15 )
    j= pd.merge(df[['DATE_TIME','HIGH','LOW']],t, how='right', on='DATE_TIME')
    x_1 = j['HIGH'].fillna(method = 'bfill')
    x_2 = j['LOW'].fillna(method = 'bfill')

    val = (x_1 - x_2)/11
    x['val'] = x['val'] + val

range_dict = dict(x.groupby('Event_type').val.mean())

y= train_df[train_df['Trend']== 'UP'].reset_index(drop=True)
y['open_close_avg'] = 0
y['high_avg'] = 0
for i in range(11):
    t= pd.DataFrame({})
    t['DATE_TIME']= y['DATE_TIME'] + pd.Timedelta(minutes = (-45) + (i+1)*15 )
    k= pd.merge(df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE']],t, how='right', on='DATE_TIME')
    y_1 = k['CLOSE'].fillna(method = 'bfill')
    y_2 = k['OPEN'].fillna(method = 'bfill')
    y_3 = k['HIGH'].fillna(method = 'bfill')

    open_close_avg = (y_1 + y_2)/2
    y['high_avg'] = y['high_avg'] +  y_3
    y['open_close_avg'] = y['open_close_avg'] + open_close_avg

y['strength'] = (y['high_avg'])/11 - (y['open_close_avg'])/11
y= y.set_index('DATE_TIME')

z= train_df[train_df['Trend']== 'DOWN'].reset_index(drop=True)
z['open_close_avg'] = 0
z['low_avg'] = 0
for i in range(11):
    t= pd.DataFrame({})
    t['DATE_TIME']= z['DATE_TIME'] + pd.Timedelta(minutes = (-45) + (i+1)*15 )
    l= pd.merge(df[['DATE_TIME','HIGH','LOW','OPEN','CLOSE']],t, how='right', on='DATE_TIME')
    z_1 = l['CLOSE'].fillna(method = 'bfill')
    z_2 = l['OPEN'].fillna(method = 'bfill')
    z_3 = l['HIGH'].fillna(method = 'bfill')

    open_close_avg = (z_1 + z_2)/2
    z['low_avg'] = z['low_avg'] +  z_3
    z['open_close_avg'] = z['open_close_avg'] + open_close_avg

z['strength'] = (z['low_avg'])/11 - (z['open_close_avg'])/11
z= z.set_index('DATE_TIME')

strength = []

for i in range(len(train_df)):
    if train_df['Trend'][i] == 'DOWN' :
        strength.append(z['strength'][train_df['DATE_TIME'][i]])
    elif train_df['Trend'][i] == 'UP' :
        strength.append(y['strength'][train_df['DATE_TIME'][i]])

    else :
        strength.append(range_dict[train_df['Event_type'][i]])
train_df['strength'] = strength
train_df
