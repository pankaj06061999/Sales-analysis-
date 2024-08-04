#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


df= pd.read_csv('.\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data\\Sales_April_2019.csv')
files = [file for file in os.listdir('.\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data')]
for file in files:
    print(file)


# In[3]:


df= pd.read_csv('.\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data\\Sales_April_2019.csv')
files = [file for file in os.listdir('.\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data')]

all_months_data = pd.DataFrame()

for file in files:
    df= pd.read_csv(".\\Pandas-Data-Science-Tasks-master\\SalesAnalysis\\Sales_Data\\"+file)
    all_months_data= pd.concat([all_months_data,df])
    
all_months_data.to_csv("all_data.csv",index=False)


# In[4]:


all_data=pd.read_csv('all_data.csv')


# aougment data with additional columns

# In[5]:


all_data.dropna()


# In[6]:


all_data['Month']= all_data['Order Date'].str[0:2]
all_data.head()


# # clean up the  data !
# 

# In[7]:


nan_df=all_data[all_data.isna().any(axis=1)]
nan_df.head()
all_data=all_data.dropna(how='all')
all_data.head()


# # find or and delete it

# In[8]:


t_df= all_data[all_data['Order Date'].str[0:2]=='Or']
t_df


# In[9]:


all_data=all_data[all_data['Order Date'].str[0:2]!='Or']


# In[10]:


# convert column to the correcr type


# In[11]:


all_data['Quantity Ordered']= pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each']=pd.to_numeric(all_data['Price Each'])
all_data.head()


# In[ ]:





# In[ ]:





# In[12]:


all_data['Month']= all_data['Order Date'].str[0:2]
all_data['Month']=all_data['Month'].astype('int32')
all_data.head()


# In[13]:


### add a sales column 


# In[14]:


all_data['Sales']=all_data['Quantity Ordered']* all_data['Price Each']
all_data.head()


# In[15]:


## ADD A CITY COLUMN


# In[16]:


# lets use apply method .apply()
all_data['city']= all_data['Purchase Address'].apply(lambda x: x.split(',')[1])
all_data.head()


# In[17]:


def get_city(address):
    # Split the address by comma and take the second part
    return address.split(',')[1].strip()

def get_state(address):
    # Split the address by comma, take the third part, and split it by space to get the state
    return address.split(',')[2].strip().split(' ')[0]

all_data['city1'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' (' + get_state(x) + ')')
all_data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Question 1 : What was the best month for sales ? how much was earned that month ?

# In[18]:


results=all_data.groupby('Month').sum()



# In[19]:


import matplotlib.pyplot as plt 
month = range (1,13)





plt.bar(month,results['Sales'])
plt.ylabel('sales in usd$')
plt.xlabel('month number')
plt.xticks(month)

plt.show()


# # question2: What city had the highest number  of sales?

# In[ ]:


results=all_data.groupby('city1').sum()


# In[ ]:


results.head()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt 
cities =[city1 for city1,df in all_data.groupby('city1')]

plt.bar (cities,results['Sales'])

plt.ylabel('sales in usd$')
plt.xlabel('citys')
plt.xticks(cities, rotation='vertical',size=8)

plt.show()


# In[ ]:


# another methode we can relate 
import matplotlib.pyplot as plt 

cities =all_data['city1'].unique()

plt.bar (cities,results['Sales'])

plt.ylabel('sales in usd$')
plt.xlabel('citys')
plt.xticks(cities, rotation='vertical',size=8)

plt.show()


# # questiom3: what time should we display advertisements to maximize likehood of cistomer's buying product 

# In[ ]:


all_data.head()


# In[ ]:


all_data.head()


# In[ ]:


import pandas as pd

# Assuming all_data is your DataFrame
# Correctly parse the 'Order Date' column with specified format
all_data['Order Date'] = pd.to_datetime(all_data['Order Date'], format='%m/%d/%y %H:%M')

# Print to verify
all_data.head()


# In[ ]:


all_data['Order Date'] = pd.to_datetime(all_data['Order Date']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


all_data['hour']= all_data['Order Date'].dt.hour
all_data['minute']=all_data['Order Date'].dt.minute
all_data.head()


# In[ ]:


hours = [hour for hour,df in all_data.groupby('hour')]
plt.plot(hours,all_data.groupby(['hour']).count())

plt.xticks(hours)
plt.xlabel('hours')
plt.ylabel('number of order')
plt.grid()
plt.show()
##  ans = 11am  and 7pm (19)


# # Question4: what product are most often sold together

# In[ ]:


all_data


# In[ ]:


df= all_data[all_data['Order ID'].duplicated(keep=False)]
df['Grouped']=df.groupby('Order ID')['Product'].transform(lambda x:  ','.join(x))
df = df[['Order ID','Grouped']].drop_duplicates()

df.head(50)


# In[ ]:


from itertools import combinations
from collections import Counter

count = Counter()
for row in df['Grouped']:
    row_list = row.split(',')
    
    count.update(Counter(combinations(row_list,2)))


for key,value in count.most_common(10):
    print(key, value)









 


# In[ ]:





# In[ ]:





# # question5: what product sold the most? why do you think it siold the most?

# In[ ]:


df=all_data.drop('Order Date',axis =1)


# In[ ]:


df


# In[ ]:


pg=df.groupby("Product")
quantity_ordered=pg.sum()['Quantity Ordered']

products = [product for product, df in pg ]
plt.bar (products, quantity_ordered)
plt.ylabel('quantity ordered')
plt.xlabel('product')

plt.xticks(products, rotation='vertical' ,  size=8)
plt.show()









# In[ ]:





# In[ ]:


nf=df.drop(['Purchase Address','city','city1'],axis=1)


# In[ ]:


nf


# In[ ]:


prices = nf.groupby('Product').mean()['Price Each']
print(prices)


# In[ ]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




