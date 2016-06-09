
# coding: utf-8

# In[2]:

import KFoldPCA
import f1_score
import pandas as pd
import numpy as np
import xgb_ensemble_final as xgb
test = pd.read_csv("test_clean.csv")
train = pd.read_csv("train_clean.csv")
train_label = train['Active_Customer']
test_na = pd.read_csv("test_clean_na.csv")
train_na = pd.read_csv("train_clean_na.csv")
train_na_label = train_na['Active_Customer']

train_ID_label = train[["Cust_id", "Active_Customer"]]
train_na_ID_label = train_na[["Cust_id", "Active_Customer"]]
# In[3]:

#### ensemble stage 1
base_models = [("xgb_tree",xgb.xgb_big_tree),("linear_tree",xgb.xgb_big_linear),               ("Random_forest",xgb.Random_forest_big),("Adaboost",xgb.Adaboost_big),               ("Extra_tree",xgb.Extra_tree_big),("Gradient_boosting",xgb.Gradient_boosting_big),              ("Linear_SVM",xgb.Linear_SVM_big),("Naive_Bayes",xgb.Naive_Bayes_big),("KNN1_big",xgb.KNN1_big),("Neural_Network_1",xgb.Neural_Network_1_big), ("Neural_Network_2",xgb.Neural_Network_2_big), ("Logistic_Regression_big", xgb.Logistic_Regression_big), ("KNN2", xgb.KNN2_big)]
base_models_na = [("xgb_tree",xgb.xgb_na_tree),("xgb_linear",xgb.xgb_na_linear),                  ("Random_forest",xgb.Random_forest_na),("Adaboost",xgb.Adaboost_na),                  ("Extra_tree",xgb.Extra_tree_na),("Gradient_boosting",xgb.Gradient_boosting_na),                 ("Linear_SVM",xgb.Linear_SVM_na),("Naive_Bayes",xgb.Naive_Bayes_na),("KNN1_na",xgb.KNN1_na), ("Logistic_na", xgb.Logistic_Regression_na)]
fold,seed = 5,2
train_list,vaild_list = KFoldPCA.generate_kfold(train,3,fold)
train_na_list,vaild_na_list = KFoldPCA.generate_kfold(train_na,3,fold)
ensemble_L1_test = pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
ensemble_L1_test_na = pd.DataFrame(test_na["Cust_id"].values,columns=["Cust_id"])
ensemble_L1_train = pd.DataFrame(train["Cust_id"].values,columns=["Cust_id"])
ensemble_L1_train_na = pd.DataFrame(train_na["Cust_id"].values,columns=["Cust_id"])


# In[ ]:

#big data
cv_score_list = []
for func_name,func in base_models:
    print (func_name,"start")
    test_result,vaild_pred, cv_score = func(train_list,vaild_list,test,seed)
    cv_score_list.append(cv_score)
    test_result=test_result.rename(columns = {'Active_Customer':func_name})
    vaild_pred=vaild_pred.rename(columns = {'Active_Customer':func_name})
    ensemble_L1_test = pd.merge(ensemble_L1_test, test_result,left_index=True, on='Cust_id')
    ensemble_L1_train = pd.merge(ensemble_L1_train, vaild_pred,left_index=True, on='Cust_id')
    print (func_name,"end")
print (sum(cv_score_list)/len(cv_score_list))


ensemble_L1_train = pd.merge(ensemble_L1_train, train_ID_label, left_index = True, on='Cust_id')

# In[ ]:

ensemble_L1_test.to_csv('ensemble_L1_test.csv',index=False)
ensemble_L1_train.to_csv('ensemble_L1_train.csv',index=False)

X_train = train.drop("Active_Customer", axis = 1)
ensemble_L1_train = pd.merge(ensemble_L1_train, X_train, left_index = True, on='Cust_id')

X_test = test
ensemble_L1_test = pd.merge(ensemble_L1_test, X_test, left_index = True, on='Cust_id')




# In[ ]:

#small
cv_score_na_list = []
for func_name,func in base_models_na:
    print (func_name,"start")
    test_result_na,vaild_pred_na, cv_score_na = func(train_na_list,vaild_na_list,test_na,seed)
    cv_score_na_list.append(cv_score_na)
    test_result_na=test_result_na.rename(columns = {'Active_Customer':func_name})
    vaild_pred_na=vaild_pred_na.rename(columns = {'Active_Customer':func_name})
    ensemble_L1_test_na = pd.merge(ensemble_L1_test_na, test_result_na,left_index=True, on='Cust_id')
    ensemble_L1_train_na = pd.merge(ensemble_L1_train_na, vaild_pred_na,left_index=True, on='Cust_id')
    print (func_name,"end")
print (sum(cv_score_na_list)/len(cv_score_na_list))


# In[ ]:

ensemble_L1_train_na = pd.merge(ensemble_L1_train_na, train_na_ID_label, left_index = True, on='Cust_id')
# ensemble_L1_train_na['Active_Customer'] = train_label
ensemble_L1_test_na.to_csv('ensemble_L1_test_na.csv',index=False)
ensemble_L1_train_na.to_csv('ensemble_L1_train_na.csv',index=False)

X_train_na = train_na.drop("Active_Customer", axis = 1)

ensemble_L1_train_na = pd.merge(ensemble_L1_train_na, X_train_na, left_index = True, on='Cust_id')

X_test_na = test_na
ensemble_L1_test_na = pd.merge(ensemble_L1_test_na, X_test_na, left_index = True, on='Cust_id')
# In[ ]:

### ensemble stage 2
#ensemble_L1_train_na['Active_Customer'] = train_label
#ensemble_L1_train['Active_Customer'] = train_na_label

ensemble_models = [("nn",xgb.Neural_Network_big_ensemble)]
ensemble_models_na = [("Active_Customer",xgb.xgb_na_tree_ensemble)]
fold,seed = 5,2
ensemble_train_list,ensemble_vaild_list = KFoldPCA.generate_kfold(ensemble_L1_train,3,fold)
ensemble_train_na_list,ensemble_vaild_na_list = KFoldPCA.generate_kfold(ensemble_L1_train_na,3,fold)
ensemble_L2_test = pd.DataFrame(ensemble_L1_test["Cust_id"].values,columns=["Cust_id"])
ensemble_L2_test_na = pd.DataFrame(ensemble_L1_test_na["Cust_id"].values,columns=["Cust_id"])
ensemble_L2_train = pd.DataFrame(ensemble_L1_train["Cust_id"].values,columns=["Cust_id"])
ensemble_L2_train_na = pd.DataFrame(ensemble_L1_train_na["Cust_id"].values,columns=["Cust_id"])


# In[ ]:

#big data
cv_score_list = []
for func_name,func in ensemble_models:
    test_result,vaild_pred, cv_score = func(ensemble_train_list,ensemble_vaild_list,ensemble_L1_test                                            ,seed)
    cv_score_list.append(cv_score)
    test_result=test_result.rename(columns = {'Active_Customer':func_name})
    vaild_pred=vaild_pred.rename(columns = {'Active_Customer':func_name})
    ensemble_L2_test = pd.merge(ensemble_L2_test, test_result,left_index=True, on='Cust_id')
    ensemble_L2_train = pd.merge(ensemble_L2_train, vaild_pred,left_index=True, on='Cust_id')

print (sum(cv_score_list)/len(cv_score_list))


# In[ ]:

ensemble_L2_test.to_csv('ensemble_L2_test.csv',index=False)
ensemble_L2_train.to_csv('ensemble_L2_train.csv',index=False)


# In[ ]:

#small
cv_score_na_list = []
for func_name,func in ensemble_models_na:
    test_result_na,vaild_pred_na, cv_score_na = func(ensemble_train_na_list,ensemble_vaild_na_list,                                                     ensemble_L1_test_na,seed)
    cv_score_na_list.append(cv_score_na)
    test_result_na=test_result_na.rename(columns = {'Active_Customer':func_name})
    vaild_pred_na=vaild_pred_na.rename(columns = {'Active_Customer':func_name})
    ensemble_L2_test_na = pd.merge(ensemble_L2_test_na, test_result_na,left_index=True, on='Cust_id')
    ensemble_L2_train_na = pd.merge(ensemble_L2_train_na, vaild_pred_na,left_index=True, on='Cust_id')

print (sum(cv_score_na_list)/len(cv_score_na_list))


# In[ ]:

ensemble_L2_test_na.to_csv('ensemble_L2_test_na.csv',index=False)
ensemble_L2_train_na.to_csv('ensemble_L2_train_na.csv',index=False)


# In[ ]:

### ensemble stage 3
'''
iensemble_L2_test["Active_Customer_temp"] = 0
ensemble_L2_test["Active_Customer"] = 0
for func_name,func in ensemble_models:
    ensemble_L2_test["Active_Customer_temp"] += ensemble_L2_test[func_name]
ensemble_L2_test["Active_Customer"][ensemble_L2_test["result"] > len(ensemble_models)/2]=1

ensemble_L2_test_na["Active_Customer_temp"] = 0
ensemble_L2_test_na["Active_Customer"] = 0
for func_name,func in ensemble_models_na:
    ensemble_L2_test_na["Active_Customer_temp"] += ensemble_L2_test_na[func_name]
ensemble_L2_test_na["Active_Customer"][ensemble_L2_test_na["result"] > len(ensemble_models_na)/2]=1



# In[ ]:

test_result = pd.DataFrame(ensemble_L2_test["Cust_id"].values,columns=["Cust_id"])
test_result["Active_Customer"] = ensemble_L2_test["Active_Customer"]
test_result_na = pd.DataFrame(ensemble_L2_test_na["Cust_id"].values,columns=["Cust_id"])
test_result_na["Active_Customer"] = ensemble_L2_test_na["Active_Customer"]

test_result = pd.concat([test_result,test_result_na])
test_result.to_csv('res2.csv',index=False)
'''
