
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
test_raw = pd.read_csv("Test.csv")
test_raw["num_na"] = test_raw.isnull().sum(axis=1).values
test_result = pd.DataFrame(columns=['Cust_id','Active_Customer'])


# In[2]:

len(test_raw)
misslist = [10330,3033,9986,8303]
test_raw.ix[misslist]


# In[3]:

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


# In[4]:

#NA Value -- train
#Tenure < 0 is all 1
#Tenure NA directly droped
test_result = test_result.T
test_result[0] = [test_raw[test_raw['Cust_Tenure'].isnull()]['Cust_id'].values[0], 0]
test_result = test_result.T
test = test_raw.drop(test_raw.index[test_raw['Cust_Tenure'].isnull()])

test_1 = test[test["Cust_Tenure"] < 0]
test_1_result = pd.DataFrame(test_1["Cust_id"].values,columns=['Cust_id'])
test_1_result['Active_Customer']=1
test_result = pd.concat([test_result,test_1_result])
test = test[test["Cust_Tenure"] >= 0]


# In[5]:

len(test)


# In[6]:

#NA Value -- train
#Tenure < 0 is all 1
#Tenure NA directly droped
# Convert Categorical Variable
if 'Cust_status' in test.columns:
        test = test.drop('Cust_status', 1)
category_list = ["Trans24","Trans25","Trans26","Trans27"]
for i in category_list:
    test[i] = test[i].apply(change_trans)
#Food NA 830 take out  (Transaction 12, 13, 14, 15)
test_na = test[test['Food1'].isnull()]
test = test[test['Food1'].notnull()]
test['Trans11_NA'] = test['Trans11'].isnull().astype(int)
#trans 11 still has 1145 NA
test["Trans10"] = test["Trans10"].fillna(41.008145)
test["Trans11"] = test["Trans11"].fillna(61.307763)
test["Trans41"] = test["Trans41"].fillna(3.021522)
#train["Trans10"].isnull().values.sum(),train["Trans11"].isnull().values.sum(),train["Trans41"].isnull().values.sum()
#Tenure scale using sqrt(x)
test["Cust_Tenure"] = test["Cust_Tenure"]**0.5


# In[7]:


# test_2_result


# In[8]:

misslist = [10330,3033,9986,8303]
test_2_result = pd.DataFrame(test.ix[misslist]["Cust_id"].values,columns=['Cust_id'])
test_2_result['Active_Customer']=0
test_2_result.set_value(1,'Active_Customer',1)
test_result = pd.concat([test_result,test_2_result])


# In[9]:

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
    test[i][test[i] < cut_below[i]] = 0
for i in drop_list:
    if i in test.columns:
        test = test.drop(i, 1)
for i in drop_above:
    if len(test[test[i] > drop_above[i]])>0:
        print (test[test[i] > drop_above[i]])
    test = test[test[i] <= drop_above[i]]
#Sqrt feature 1,2,3,4,41
for i in sqrt_feature:
    test[i] = test[i]**0.5


# In[10]:

#Food
#drop F17883, F25409, F3328, F10213
#drop_list = ["F17883", "F25409","F3328", "F10213"]
#for i in drop_list:
#    train = train.drop(train.index[train["Cust_id"]==i])


# In[11]:

len(test)


# In[12]:

#Promotion
#drop F12522, Promotion 12 = 7
#test = test[test["Promotion12"] <= 7]
#drop promotion 37
if "Promotion37" in test.columns:
    test = test.drop("Promotion37", 1)
#the promotion 5,6,7,8,13,21 cateogrical
cut_above = ["Promotion5","Promotion6","Promotion7","Promotion8","Promotion13","Promotion21"]
for i in cut_above:
    test[i][test[i] > 1]=1
#drop cateogrical keep "Trans26"         "Trans27"         "Promotion8"
drop_list = ["Promotion5","Promotion6","Promotion7","Promotion13","Promotion21","Cust_status",            "Trans24","Trans25"]
for i in drop_list:
    if i in test.columns:
        test = test.drop(i, 1)
#rename cateogrical columns
rename_list = ["Trans26","Trans27","Promotion8","Trans11_NA","num_na"]
count = 0
for i in rename_list:
    if i in test.columns:
        temp = test[i]
        test = test.drop(i, 1)
        test = pd.concat([test,temp], axis=1)
test=test.rename(columns = {"Trans26":"cate1"})
test=test.rename(columns = {"Trans27":"cate2"})
test=test.rename(columns = {"Promotion8":"cate3"})
test=test.rename(columns = {"Trans11_NA":"cate4"})


# In[13]:

#save
test["num_0"] = (test == 0).astype(int).sum(axis=1).values
temp_test_trans = test[test.columns[2:35]]
temp_test_food = test[test.columns[35:199]]
temp_test_promo = test[test.columns[199:240]]
test["num_0_trans"] = (temp_test_trans == 0).astype(int).sum(axis=1).values
test["num_0_food"] = (temp_test_food == 0).astype(int).sum(axis=1).values
test["num_0_promo"] = (temp_test_promo == 0).astype(int).sum(axis=1).values
save_test = test
save_test.to_csv('test_clean.csv')


# In[14]:

test_na_set = test_na.reset_index()
test_na_set = test_na_set.drop("index", 1)
test_na_set = pd.concat([test_na_set[test_na_set.columns[:43]],test_na_set[test_na_set.columns[207:]]], axis=1)
test_na_set["num_na"] = test_na_set.isnull().sum(axis=1).values
test_na_set['NA_cate1'] = test_na_set['Trans12'].isnull().astype(int)
test_na_set['NA_cate2'] = test_na_set['Trans10'].isnull().astype(int)
test_na_set['NA_cate3'] = test_na_set['Trans11'].isnull().astype(int)
test_na_set['NA_cate4'] = test_na_set['Trans41'].isnull().astype(int)
drop_list = ["Trans10","Trans11","Trans12","Trans13","Trans14","Trans15","Trans41"]
for i in drop_list:
    if i in test_na_set.columns:
        test_na_set = test_na_set.drop(i, 1)


# In[15]:

drop_above = {"Trans1":1000,"Trans2":1000,"Trans3":1000,"Trans4":1000,
            "Trans5":100,"Trans6":100}
cut_below = {"Trans1":0,"Trans2":0,"Trans3":0,"Trans4":0}
drop_list = ["Trans20","Trans21","Trans22","Trans23"]
sqrt_feature = ["Trans1","Trans2","Trans3","Trans4"]
for i in cut_below:
    test_na_set[i][test_na_set[i] < cut_below[i]]=0
for i in drop_list:
    if i in test_na_set.columns:
        test_na_set = test_na_set.drop(i, 1)
for i in drop_above:
    if len(test_na_set[test_na_set[i] > drop_above[i]])>0:
        print (test_na_set[test_na_set[i] > drop_above[i]])
    test_na_set = test_na_set[test_na_set[i] <= drop_above[i]]
#Sqrt feature 1,2,3,4,41
for i in sqrt_feature:
    test_na_set[i] = test_na_set[i]**0.5


# In[16]:

#drop promotion 37
if "Promotion37" in test_na_set.columns:
    test_na_set = test_na_set.drop("Promotion37", 1)
#the promotion 5,6,7,8,13,21 cateogrical
cut_above = ["Promotion5","Promotion6","Promotion7","Promotion8","Promotion13","Promotion21"]
for i in cut_above:
    test_na_set[i][test_na_set[i] > 1]=1
#drop cateogrical keep "Trans26"         "Trans27"         "Promotion8"
drop_list = ["Promotion5","Promotion6","Promotion7","Promotion13","Promotion21","Cust_status",            "Trans24","Trans25"]
for i in drop_list:
    if i in test_na_set.columns:
        test_na_set = test_na_set.drop(i, 1)
#rename cateogrical columns
rename_list = ["Trans26","Trans27","Promotion8","num_na"]
count = 0
for i in rename_list:
    if i in test_na_set.columns:
        temp = test_na_set[i]
        test_na_set = test_na_set.drop(i, 1)
        test_na_set = pd.concat([test_na_set,temp], axis=1)
test_na_set=test_na_set.rename(columns = {"Trans26":"NA_cate5"})
test_na_set=test_na_set.rename(columns = {"Trans27":"NA_cate6"})
test_na_set=test_na_set.rename(columns = {"Promotion8":"NA_cate7"})


# In[17]:

#save
test_na_set["num_0"] = (test_na_set == 0).astype(int).sum(axis=1).values
temp_test_trans_na = test_na_set[test_na_set.columns[2:28]]
temp_test_promo_na = test_na_set[test_na_set.columns[29:69]]
test_na_set["num_0_trans"] = (temp_test_trans_na == 0).astype(int).sum(axis=1).values
test_na_set["num_0_food"] = (temp_test_promo_na == 0).astype(int).sum(axis=1).values
save_test_na = test_na_set
save_test_na.to_csv('test_clean_na.csv')


# In[18]:

test_na_set.columns


# In[19]:

len(test_na_set),len(test)


# In[20]:

test_result


# In[21]:

###########################################################
test_result_3 = pd.read_csv("ensemble_L2_test.csv")
test_result_4 = pd.read_csv("ensemble_L2_test_na.csv")
#######################################


# In[22]:

test_result = pd.concat([test_result,test_result_3,test_result_4])


# In[23]:

test_result


# In[24]:

len(test_result)


# In[25]:

test_id = pd.read_csv("Test.csv")
test_id = pd.DataFrame(test_id["Cust_id"].values,columns=["Cust_id"])


# In[26]:

out = pd.merge(test_id, test_result,left_index=True, on='Cust_id')
out.to_csv("submit_19.csv", index=False)


# In[ ]:



