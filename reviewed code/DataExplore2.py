
# coding: utf-8

# In[1]:

# Convert Categorical Variable
def change_cust_status(x):
    if x == 'Old':
        return 0
    elif x == 'New':
        return 1
def change_trans(x):
    if x == 'Enable':
        return 1
    elif x == 'Not-Enable':
        return 0  


# In[2]:

import pandas as pd
import numpy as np
train_raw = pd.read_csv("data/Train.csv")


# In[3]:

train_raw["num_na"] = train_raw.isnull().sum(axis=1).values


# In[4]:

#NA Value -- train 
#Tenure < 0 is all 1
#Tenure NA directly droped
train = train_raw.drop(train_raw.index[train_raw['Cust_Tenure'].isnull()])
train_1 = train[train["Cust_Tenure"] < 0]
train = train[train["Cust_Tenure"] >= 0]
# Convert Categorical Variable
#train['Cust_status'] = train['Cust_status'].apply(change_cust_status)
if 'Cust_status' in train.columns:
        train = train.drop('Cust_status', 1) 
category_list = ["Trans24","Trans25","Trans26","Trans27"]
for i in category_list:
    train[i] = train[i].apply(change_trans)
#Food NA 830 take out  (Transaction 12, 13, 14, 15)
train_na = train[train['Food1'].isnull()]
train = train[train['Food1'].notnull()]
train['Trans11_NA'] = train['Trans11'].isnull().astype(int)
#trans 11 still has 1145 NA
train = train.fillna(train.mean()["Trans10":"Trans11"])
train = train.fillna(train.mean()["Trans41":"Trans41"])
#train["Trans10"].isnull().values.sum(),train["Trans11"].isnull().values.sum(),train["Trans41"].isnull().values.sum()
#Tenure scale using sqrt(x)
train["Cust_Tenure"] = train["Cust_Tenure"]**0.5


# In[5]:

#Trans
#Transaction 1,2,3,4 cut the value above 1000 F36097, if t <0 then t = 0, 
#Transaction 5, 6 cut the value abvoe 100
#Transaction 20,21,22,23 drop
drop_above = {"Trans1":1000,"Trans2":1000,"Trans3":1000,"Trans4":1000,
            "Trans5":100,"Trans6":100}
cut_below = {"Trans1":0,"Trans2":0,"Trans3":0,"Trans4":0}
drop_list = ["Trans20","Trans21","Trans22","Trans23"]
sqrt_feature = ["Trans1","Trans2","Trans3","Trans4","Trans41"]
for i in cut_below:
    train[i][train[i] < cut_below[i]]=0
for i in drop_list:
    if i in train.columns:
        train = train.drop(i, 1)  
for i in drop_above:
    train = train[train[i] <= drop_above[i]]
#Sqrt feature 1,2,3,4,41
for i in sqrt_feature:
    train[i] = train[i]**0.5


# In[6]:

#Food
drop_list = ['F17883','F25409','F3328', 'F10213'] 
for i in drop_list:
    train = train.drop(train.index[train["Cust_id"]==i])


# In[7]:

#Promotion
#drop F12522, Promotion 12 = 7
train = train[train["Promotion12"] <= 7]
#drop promotion 37
if "Promotion37" in train.columns:
    train = train.drop("Promotion37", 1)
#the promotion 5,6,7,8,13,21 cateogrical 
cut_above = ["Promotion5","Promotion6","Promotion7","Promotion8","Promotion13","Promotion21"]
for i in cut_above:
    train[i][train[i] > 1]=1
#drop cateogrical keep "Trans26"         "Trans27"         "Promotion8"
drop_list = ["Promotion5","Promotion6","Promotion7","Promotion13","Promotion21","Cust_status",            "Trans24","Trans25"]
for i in drop_list:
    if i in train.columns:
        train = train.drop(i, 1) 
#rename cateogrical columns
rename_list = ["Trans26","Trans27","Promotion8","Trans11_NA","num_na"]
count = 0
for i in rename_list:
    if i in train.columns:
        temp = train[i]
        train = train.drop(i, 1)
        train = pd.concat([train,temp], axis=1)
train=train.rename(columns = {"Trans26":"cate1"})
train=train.rename(columns = {"Trans27":"cate2"})
train=train.rename(columns = {"Promotion8":"cate3"})
train=train.rename(columns = {"Trans11_NA":"cate4"})


# In[8]:

train["num_0"] = (train == 0).astype(int).sum(axis=1).values
temp_train_trans = train[train.columns[2:35]]
temp_train_food = train[train.columns[35:199]]
temp_train_promo = train[train.columns[199:240]]


# In[9]:

train["num_0_trans"] = (temp_train_trans == 0).astype(int).sum(axis=1).values
train["num_0_food"] = (temp_train_food == 0).astype(int).sum(axis=1).values
train["num_0_promo"] = (temp_train_promo == 0).astype(int).sum(axis=1).values


# In[10]:

#save
train_lab = train["Active_Customer"]
save_train = train.drop("Active_Customer", 1)
save_train = pd.concat([save_train,train_lab], axis=1)
save_train.to_csv('train_clean.csv')


# In[11]:

#Food NA data
train_na_set = train_na.reset_index()
train_na_set = train_na_set.drop("index", 1)
train_na_set = pd.concat([train_na_set[train_na_set.columns[:43]],train_na_set[train_na_set.columns[207:]]], axis=1)
train_na_set["num_na"] = train_na_set.isnull().sum(axis=1).values
train_na_set['NA_cate1'] = train_na_set['Trans12'].isnull().astype(int)
train_na_set['NA_cate2'] = train_na_set['Trans10'].isnull().astype(int)
train_na_set['NA_cate3'] = train_na_set['Trans11'].isnull().astype(int)
train_na_set['NA_cate4'] = train_na_set['Trans41'].isnull().astype(int)
drop_list = ["Trans10","Trans11","Trans12","Trans13","Trans14","Trans15","Trans41"]
for i in drop_list:
    if i in train_na_set.columns:
        train_na_set = train_na_set.drop(i, 1)  



# In[12]:

drop_above = {"Trans1":1000,"Trans2":1000,"Trans3":1000,"Trans4":1000,
            "Trans5":100,"Trans6":100}
cut_below = {"Trans1":0,"Trans2":0,"Trans3":0,"Trans4":0}
drop_list = ["Trans20","Trans21","Trans22","Trans23"]
sqrt_feature = ["Trans1","Trans2","Trans3","Trans4"]
for i in cut_below:
    train_na_set[i][train_na_set[i] < cut_below[i]]=0
for i in drop_list:
    if i in train_na_set.columns:
        train_na_set = train_na_set.drop(i, 1)  
for i in drop_above:
    train_na_set = train_na_set[train_na_set[i] <= drop_above[i]]
#Sqrt feature 1,2,3,4,41
for i in sqrt_feature:
    train_na_set[i] = train_na_set[i]**0.5


# In[13]:

#drop promotion 37
if "Promotion37" in train_na_set.columns:
    train_na_set = train_na_set.drop("Promotion37", 1)
#the promotion 5,6,7,8,13,21 cateogrical 
cut_above = ["Promotion5","Promotion6","Promotion7","Promotion8","Promotion13","Promotion21"]
for i in cut_above:
    train_na_set[i][train_na_set[i] > 1]=1
#drop cateogrical keep "Trans26"         "Trans27"         "Promotion8"
drop_list = ["Promotion5","Promotion6","Promotion7","Promotion13","Promotion21","Cust_status",            "Trans24","Trans25"]
for i in drop_list:
    if i in train_na_set.columns:
        train_na_set = train_na_set.drop(i, 1) 
#rename cateogrical columns
rename_list = ["Trans26","Trans27","Promotion8","num_na"]
count = 0
for i in rename_list:
    if i in train_na_set.columns:
        temp = train_na_set[i]
        train_na_set = train_na_set.drop(i, 1)
        train_na_set = pd.concat([train_na_set,temp], axis=1)
train_na_set=train_na_set.rename(columns = {"Trans26":"NA_cate5"})
train_na_set=train_na_set.rename(columns = {"Trans27":"NA_cate6"})
train_na_set=train_na_set.rename(columns = {"Promotion8":"NA_cate7"})


# In[14]:

train_na_set["num_0"] = (train_na_set == 0).astype(int).sum(axis=1).values
temp_train_trans_na = train_na_set[train_na_set.columns[2:28]]
temp_train_promo_na = train_na_set[train_na_set.columns[29:69]]
train_na_set["num_0_trans"] = (temp_train_trans_na == 0).astype(int).sum(axis=1).values
train_na_set["num_0_food"] = (temp_train_promo_na == 0).astype(int).sum(axis=1).values


# In[15]:

#save
train_na_lab = train_na_set["Active_Customer"]
save_train_na = train_na_set.drop("Active_Customer", 1)
save_train_na = pd.concat([save_train_na,train_na_lab], axis=1)
save_train_na.to_csv('train_clean_na.csv')


# In[ ]:




# In[ ]:




# In[ ]:



