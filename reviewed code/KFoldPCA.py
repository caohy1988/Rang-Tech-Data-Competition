
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
train = pd.read_csv("train_clean.csv")


# In[2]:

def generate_kfold(train,seed,folds):
    from sklearn.cross_validation import StratifiedKFold
    train_lab = train["Active_Customer"]
    kf = StratifiedKFold(train_lab, n_folds=folds, shuffle = True, random_state=seed)
    train_list = []
    test_list = []
    for train_index, test_index in kf:
        temp_train = train.ix[train_index]
        temp_test = train.ix[test_index]
        train_list.append(temp_train)
        test_list.append(temp_test)
    return train_list,test_list


# In[3]:

#std trans
def pca_scaler_step(train):
    from sklearn import preprocessing
    trans_scaler = preprocessing.MinMaxScaler()
    trans_df = train[train.columns[3:36]]
    trans_m = trans_df.values.astype(float)
    trans_std = trans_scaler.fit_transform(trans_m)
    food_scaler = preprocessing.MinMaxScaler()
    food_df = train[train.columns[36:200]]
    food_m = food_df.values.astype(float)
    food_std = food_scaler.fit_transform(food_m)
    promo_scaler = preprocessing.MinMaxScaler()
    promo_df = train[train.columns[200:241]]
    promo_m = promo_df.values.astype(float)
    promo_std = promo_scaler.fit_transform(promo_m)
    other_scaler = preprocessing.MinMaxScaler()
    other_df = train[train.columns[245:250]]
    other_m = other_df.values.astype(float)
    other_std = other_scaler.fit_transform(other_m)
    scaler_dict = {"trans":trans_scaler,"food":food_scaler,"promo":promo_scaler,"other":other_scaler}
    return scaler_dict,trans_std,food_std,promo_std,other_std


# In[4]:

#trans pca
#Using PCA to treat the numerical variables in Transaction, use correlation matrix 
def pca_step(trans_std,food_std,promo_std):
    from sklearn.decomposition import PCA as sklearnPCA
    trans_pca = sklearnPCA(n_components=9)
    trans_new = trans_pca.fit_transform(trans_std)

    #food pca
    food_pca = sklearnPCA(n_components=24)
    food_new = food_pca.fit_transform(food_std)

    # promo PCA
    promo_pca = sklearnPCA(n_components=13)
    promo_new = promo_pca.fit_transform(promo_std)

    pca_dict = {"trans":trans_pca,"food":food_pca,"promo":promo_pca}
    return trans_new,food_new,promo_new,pca_dict


# In[5]:

#create new matrix
#"Trans26"         "Trans27"         "Promotion8"
def train_pca(train):
    scaler_dict,trans_std,food_std,promo_std,other_std = pca_scaler_step(train)
    trans_new,food_new,promo_new,pca_dict = pca_step(trans_std,food_std,promo_std)
    trans_index = []
    for i in range(len(trans_new[0])):
        trans_index.append('trans'+str(i+1))
    food_index = []
    for i in range(len(food_new[0])):
        food_index.append('food'+str(i+1))
    promo_index = []
    for i in range(len(promo_new[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(trans_new)
    trans_new_df.columns = trans_index
    food_new_df = pd.DataFrame(food_new)
    food_new_df.columns = food_index
    promo_new_df = pd.DataFrame(promo_new)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(other_std)
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_food","num_0_promo"]
    train = train.reset_index()
    new_train = pd.concat([train["Cust_id"],trans_new_df, food_new_df, promo_new_df,train[train.columns[242:246]],other_new_df], axis=1)
    new_train_label = train["Active_Customer"]
    return pca_dict,scaler_dict,new_train,new_train_label


# In[6]:

def test_scale_step(scaler_dict,test):
    from sklearn import preprocessing
    #trans
    test_trans_df = test[test.columns[3:36]]
    test_trans_m = test_trans_df.values.astype(float)
    test_trans_std = scaler_dict["trans"].transform(test_trans_m)
    test_food_df = test[test.columns[36:200]]
    test_food_m = test_food_df.values.astype(float)
    test_food_std = scaler_dict["food"].transform(test_food_m)
    test_promo_df = test[test.columns[200:241]]
    test_promo_m = test_promo_df.values.astype(float)
    test_promo_std = scaler_dict["promo"].transform(test_promo_m)
    test_other_df = test[test.columns[245:250]]
    test_other_m = test_other_df.values.astype(float)
    test_other_std = scaler_dict["other"].transform(test_other_m)
    return test_trans_std,test_food_std,test_promo_std,test_other_std     


# In[7]:

def test_pca_step(pca_dict,test_trans_std,test_food_std,test_promo_std):
    test_trans_pca = pca_dict["trans"].transform(test_trans_std)
    test_food_pca = pca_dict["food"].transform(test_food_std)
    test_promo_pca = pca_dict["promo"].transform(test_promo_std)
    return test_trans_pca,test_food_pca,test_promo_pca


# In[8]:

def test_pca(scaler_dict,pca_dict,test):
    test_trans_std,test_food_std,test_promo_std,test_other_std  = test_scale_step(scaler_dict,test)
    test_trans_pca,test_food_pca,test_promo_pca = test_pca_step(pca_dict,test_trans_std,test_food_std,test_promo_std)
    trans_index = []
    for i in range(len(test_trans_pca[0])):
        trans_index.append('trans'+str(i+1))
    food_index = []
    for i in range(len(test_food_pca[0])):
        food_index.append('food'+str(i+1))
    promo_index = []
    for i in range(len(test_promo_pca[0])):
        promo_index.append('promo'+str(i+1))
        
    trans_new_df = pd.DataFrame(test_trans_pca)
    trans_new_df.columns = trans_index
    food_new_df = pd.DataFrame(test_food_pca)
    food_new_df.columns = food_index
    promo_new_df = pd.DataFrame(test_promo_pca)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(test_other_std )
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_food","num_0_promo"]
    test = test.reset_index()
    new_test = pd.concat([test["Cust_id"],trans_new_df, food_new_df, promo_new_df,test[test.columns[242:246]],other_new_df], axis=1)
    #new_test_label = test["Active_Customer"]
    return new_test
    


# In[9]:

#调用方法
# kfold 
#train_list,test_list = generate_kfold(train,5,5)
# return:train_list,test_list含有5个element的list，每个element是一个data set

#pca
#pca_dict,scaler_dict,new_train,new_train_label=train_pca(train)
#return new_train,new_train_label是train data x，y。pca_dict,scaler_dict是scaler和pca的参数
#test
#new_test = test_pca(scaler_dict,pca_dict,test)


# In[10]:

def pca_scaler_step_na(train):
    from sklearn import preprocessing
    trans_scaler = preprocessing.MinMaxScaler()
    trans_df = train[train.columns[3:29]]
    trans_m = trans_df.values.astype(float)
    trans_std = trans_scaler.fit_transform(trans_m)
    promo_scaler = preprocessing.MinMaxScaler()
    promo_df = train[train.columns[29:70]]
    promo_m = promo_df.values.astype(float)
    promo_std = promo_scaler.fit_transform(promo_m)
    other_scaler = preprocessing.MinMaxScaler()
    other_df = train[train.columns[77:81]]
    other_m = other_df.values.astype(float)
    other_std = other_scaler.fit_transform(other_m)
    scaler_dict = {"trans":trans_scaler,"promo":promo_scaler,"other":other_scaler}
    return scaler_dict,trans_std,promo_std,other_std


# In[11]:

#trans pca
#Using PCA to treat the numerical variables in Transaction, use correlation matrix 
def pca_step_na(trans_std,promo_std):
    from sklearn.decomposition import PCA as sklearnPCA
    trans_pca = sklearnPCA(n_components=8)
    trans_new = trans_pca.fit_transform(trans_std)
  
    # promo PCA
    promo_pca = sklearnPCA(n_components=12)
    promo_new = promo_pca.fit_transform(promo_std)
    pca_dict = {"trans":trans_pca,"promo":promo_pca}
    return trans_new,promo_new,pca_dict


# In[12]:

def train_pca_na(train):
    scaler_dict,trans_std,promo_std,other_std = pca_scaler_step_na(train)
    trans_new,promo_new,pca_dict = pca_step_na(trans_std,promo_std)
    trans_index = []
    for i in range(len(trans_new[0])):
        trans_index.append('trans'+str(i+1))
    promo_index = []
    for i in range(len(promo_new[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(trans_new)
    trans_new_df.columns = trans_index
    promo_new_df = pd.DataFrame(promo_new)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(other_std)

    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_promo"]
    train = train.reset_index()
    new_train = pd.concat([train["Cust_id"],trans_new_df, promo_new_df,train[train.columns[71:78]],other_new_df], axis=1)
    new_train_label = train["Active_Customer"]
    return pca_dict,scaler_dict,new_train,new_train_label


# In[13]:

def test_scale_step_na(scaler_dict,test):
    from sklearn import preprocessing
    #trans
    test_trans_df = test[test.columns[3:29]]
    test_trans_m = test_trans_df.values.astype(float)
    test_trans_std = scaler_dict["trans"].transform(test_trans_m)
    test_promo_df = test[test.columns[29:70]]
    test_promo_m = test_promo_df.values.astype(float)
    test_promo_std = scaler_dict["promo"].transform(test_promo_m)
    test_other_df = test[test.columns[77:81]]
    test_other_m = test_other_df.values.astype(float)
    test_other_std = scaler_dict["other"].transform(test_other_m)
    return test_trans_std,test_promo_std,test_other_std     


# In[14]:

def test_pca_step_na(pca_dict,test_trans_std,test_promo_std):
    test_trans_pca = pca_dict["trans"].transform(test_trans_std)
    test_promo_pca = pca_dict["promo"].transform(test_promo_std)
    return test_trans_pca,test_promo_pca


# In[15]:

def test_pca_na(scaler_dict,pca_dict,test):
    test_trans_std,test_promo_std,test_other_std = test_scale_step_na(scaler_dict,test)
    test_trans_pca,test_promo_pca = test_pca_step_na(pca_dict,test_trans_std,test_promo_std)
    trans_index = []
    for i in range(len(test_trans_pca[0])):
        trans_index.append('trans'+str(i+1))
    promo_index = []
    for i in range(len(test_promo_pca[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(test_trans_pca)
    trans_new_df.columns = trans_index
    promo_new_df = pd.DataFrame(test_promo_pca)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(test_other_std)
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_promo"]
    test = test.reset_index()
    new_test = pd.concat([test["Cust_id"],trans_new_df, promo_new_df,test[test.columns[71:78]],other_new_df], axis=1)
    #new_test_label = test["Active_Customer"]
    return new_test


# In[16]:

#调用方法
# kfold 
#na_train_list,na_test_list = generate_kfold(train,5,10)
# return:train_list,test_list含有5个element的list，每个element是一个data set
#na_train,na_test = na_train_list[0],na_test_list[0]
#pca
#pca_dict,scaler_dict,new_train,new_train_label=train_pca_na(na_train)
#return new_train,new_train_label是train data x，y。pca_dict,scaler_dict是scaler和pca的参数
#new_test = test_pca_na(scaler_dict,pca_dict,na_test)


# In[17]:

def train_scale_all(train):
    scaler_dict,trans_new,food_new,promo_new,other_std = pca_scaler_step(train)
    trans_index = []
    for i in range(len(trans_new[0])):
        trans_index.append('trans'+str(i+1))
    food_index = []
    for i in range(len(food_new[0])):
        food_index.append('food'+str(i+1))
    promo_index = []
    for i in range(len(promo_new[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(trans_new)
    trans_new_df.columns = trans_index
    food_new_df = pd.DataFrame(food_new)
    food_new_df.columns = food_index
    promo_new_df = pd.DataFrame(promo_new)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(other_std)
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_food","num_0_promo"]
    train = train.reset_index()
    new_train = pd.concat([train["Cust_id"],trans_new_df, food_new_df, promo_new_df,train[train.columns[242:246]],other_new_df], axis=1)
    new_train_label = train["Active_Customer"]
    return scaler_dict,new_train,new_train_label
    


# In[18]:

def test_scale_all(scaler_dict,test):
    test_trans_pca,test_food_pca,test_promo_pca,test_other_std  = test_scale_step(scaler_dict,test)
    trans_index = []
    for i in range(len(test_trans_pca[0])):
        trans_index.append('trans'+str(i+1))
    food_index = []
    for i in range(len(test_food_pca[0])):
        food_index.append('food'+str(i+1))
    promo_index = []
    for i in range(len(test_promo_pca[0])):
        promo_index.append('promo'+str(i+1))
        
    trans_new_df = pd.DataFrame(test_trans_pca)
    trans_new_df.columns = trans_index
    food_new_df = pd.DataFrame(test_food_pca)
    food_new_df.columns = food_index
    promo_new_df = pd.DataFrame(test_promo_pca)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(test_other_std )
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_food","num_0_promo"]
    test = test.reset_index()
    new_test = pd.concat([test["Cust_id"],trans_new_df, food_new_df, promo_new_df,test[test.columns[242:246]],other_new_df], axis=1)
    #new_test_label = test["Active_Customer"]
    return new_test
    


# In[19]:

def train_scale_all_na(train):
    scaler_dict,trans_new,promo_new,other_std = pca_scaler_step_na(train)
    trans_index = []
    for i in range(len(trans_new[0])):
        trans_index.append('trans'+str(i+1))
    promo_index = []
    for i in range(len(promo_new[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(trans_new)
    trans_new_df.columns = trans_index
    promo_new_df = pd.DataFrame(promo_new)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(other_std)

    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_promo"]
    train = train.reset_index()
    new_train = pd.concat([train["Cust_id"],trans_new_df, promo_new_df,train[train.columns[71:78]],other_new_df], axis=1)
    new_train_label = train["Active_Customer"]
    return pca_dict,scaler_dict,new_train,new_train_label


# In[20]:

def test_scale_all_na(scaler_dict,test):
    test_trans_pca,test_promo_pca,test_other_std = test_scale_step_na(scaler_dict,test)
    trans_index = []
    for i in range(len(test_trans_pca[0])):
        trans_index.append('trans'+str(i+1))
    promo_index = []
    for i in range(len(test_promo_pca[0])):
        promo_index.append('promo'+str(i+1))
    trans_new_df = pd.DataFrame(test_trans_pca)
    trans_new_df.columns = trans_index
    promo_new_df = pd.DataFrame(test_promo_pca)
    promo_new_df.columns = promo_index 
    other_new_df = pd.DataFrame(test_other_std)
    other_new_df.columns = ["num_na","num_0","num_0_trans","num_0_promo"]
    test = test.reset_index()
    new_test = pd.concat([test["Cust_id"],trans_new_df, promo_new_df,test[test.columns[71:78]],other_new_df], axis=1)
    #new_test_label = test["Active_Customer"]
    return new_test


# In[21]:

#######scale all #####
#scaler_dict,new_train,new_train_label = train_scale_all(train)
#new_test = test_scale_all(scaler_dict,test)

