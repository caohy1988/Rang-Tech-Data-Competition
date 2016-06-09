
# coding: utf-8

# In[ ]:

'''
For big data
First layer model list
1. xgb tree classifier
2. xgb linear classifier with pca
3. extra tree classifier
4. random forest classifier with pca
5. Adaboost classifier
6. graidient boost classifier with pca
7. svm with pca
8. naive bayes with pca
9. 2 layer neural network with pca
10. 3 layer neural netork with pca
11. logistic regression with pca
12. KNN with lasso with pca
13. Total KNN

Second layer model list
1. xgb tree
2. 3 layer neural network
3. Total KNN
'''

'''
For small data
First layer model list
1. xgb tree classifier
2. xgb linear classifier with pca
3. extra tree classifier
4. random forest classifier with pca
5. Adaboost classifier
6. graidient boost classifier with pca
7. svm with pca
8. naive bayes with pca
9. logistic regression with pca
10. KNN with lasso with pca
11. Total KNN

Second layer model list
1. xgb tree
2. Total KNN
'''


# In[1]:

# xgb tree big data
#def xgb_big_tree(train_list,valid_list,test,booster,seed):
def xgb_big_tree(train_list,valid_list,test,seed):
    import xgboost as xgb
    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    x_parameters = {"max_depth":8, "objective": "multi:softmax",
                    'silent':1, "eval_metric": "error", "num_class":2,
                    "learning_rate": 0.1, 'silent':1, "min_child_weight": 0.5,
                    "subsample": 0.8,"seed":seed,"eta":0.2, "colsample_bytree":0.9,"colsample_bylevel":0.9}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    print (cv_score)

    return test_result, valid_pred, cv_score


# In[2]:

#xgb tree small data
#def xgb_na(train_list,valid_list,test,booster,seed):
def xgb_na_tree(train_list,valid_list,test,seed):
    import xgboost as xgb
    import f1_score
    import pandas as pd
    import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #            'silent':1, "eval_metric": "error", "num_class":2,
    #            "lambda":10, "seed":5, "lambda_bias": 10}
    #if booster == "tree":
    x_parameters = {"max_depth":3, "objective": "multi:softmax", "num_class":2, "learning_rate": 0.05, 'silent':1,                   "min_child_weight": 3,  "colsample_bytree":0.8,"colsample_bylevel":0.8,"eta":0.4}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]
    print (cv_score)

    return test_result, valid_pred, cv_score


# In[10]:

### xgb linear small data
def xgb_big_linear(train_list,valid_list,test,seed):
    import xgboost as xgb
    import f1_score
    import pandas as pd
    import numpy as np

    x_parameters = {"booster":"gblinear","objective": "multi:softmax",
                    'silent':1, "eval_metric": "error", "num_class":2,
                    "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]
    print (cv_score)

    return test_result, valid_pred, cv_score


# In[9]:

### xgb linear big data
def xgb_na_linear(train_list,valid_list,test,seed):
    import xgboost as xgb
    import f1_score
    import pandas as pd
    import numpy as np

    x_parameters = {"booster":"gblinear","objective": "multi:softmax",
                    'silent':1, "eval_metric": "error", "num_class":2,
                    "lambda":1, "seed":seed, "lambda_bias": 0}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[19]:

def Extra_tree_big(train_list,valid_list,test,seed):
    from sklearn.ensemble import ExtraTreesClassifier

    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    n_estimator = 1000
    max_depth = 8
    min_samples_split = 400
    min_samples_leaf = 120
    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)
        extr = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimator,max_depth = max_depth,min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, verbose = 1)
        extr.fit(X_train_noid, train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        y_valid = extr.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += extr.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]
    print (cv_score)
    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[24]:

def Extra_tree_na(train_list,valid_list,test,seed):
    from sklearn.ensemble import ExtraTreesClassifier

    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    n_estimator = 400
    max_depth = 5
    min_samples_split = 10
    min_samples_leaf = 10
    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)
        extr = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimator,max_depth = max_depth,min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, verbose = 1)
        extr.fit(X_train_noid, train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        y_valid = extr.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += extr.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]
    print (cv_score)
    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[30]:

### Random Forest Classifier big with 650 classifier
def Random_forest_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import RandomForestClassifier
    n_estimator = 1000

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimator, verbose = 1)
        rf.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = rf.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + rf.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[34]:

### Random Forest Classifier na with 700 classifier
def Random_forest_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import RandomForestClassifier
    n_estimator = 700

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        rf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimator, verbose = 1)
        rf.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = rf.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + rf.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]
    print (cv_score)

    return test_result, valid_pred, cv_score


# In[ ]:

def Adaboost_big(train_list,valid_list,test,seed):
    from sklearn.ensemble import AdaBoostClassifier

    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    n_estimator = 200
    max_depth = 6
    min_samples_split = 400
    min_samples_leaf = 120
    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)
        #extr = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimator,max_depth = max_depth,min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, verbose = 1)
        adb = sklearn.ensemble.AdaBoostClassifier(n_estimators = n_estimator)
        adb.fit(X_train_noid, train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        y_valid = adb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += adb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[41]:

def Adaboost_na(train_list,valid_list,test,seed):
    from sklearn.ensemble import AdaBoostClassifier

    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    n_estimator = 7
    max_depth = 6
    min_samples_split = 400
    min_samples_leaf = 120
    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)
        #extr = sklearn.ensemble.ExtraTreesClassifier(n_estimators = n_estimator,max_depth = max_depth,min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split, verbose = 1)
        adb = sklearn.ensemble.AdaBoostClassifier(n_estimators = n_estimator)
        adb.fit(X_train_noid, train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        y_valid = adb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += adb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]
    print (cv_score)

    return test_result, valid_pred, cv_score


# In[47]:

def Gradient_boosting_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import GradientBoostingClassifier
    n_estimator = 800
    min_samples_split = 800
    min_samples_leaf = 70
    max_depth = 6
    subsample = 0.6
    learning_rate = 0.01
    random_state = seed

    feature_list = np.array(['num_0','num_0_food','num_0_promo','num_0_trans','trans3','trans4','cate2','food22','cate3','num_na','food19','trans6',
 'promo9','trans2','promo8','food10','food11','promo12','food16','promo11','trans5','food23','promo6','food18',
 'food13','cate1','food17','food3','promo2','food14','food21','food9','promo10','food8'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        X_train_noid = X_train_noid[feature_list]
        X_valid_noid = X_valid_noid[feature_list]
        X_test_noid = X_test_noid[feature_list]

        gdb = sklearn.ensemble.GradientBoostingClassifier(n_estimators = n_estimator, learning_rate = learning_rate, max_depth = max_depth, subsample=subsample, random_state = random_state,min_samples_split=min_samples_split,min_samples_leaf = min_samples_leaf,verbose = 1)


        gdb.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = gdb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + gdb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[48]:

def Gradient_boosting_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import GradientBoostingClassifier
    n_estimator = 15
    min_samples_split = 200
    min_samples_leaf = 50
    max_depth = 8
    subsample = 0.8
    learning_rate = 0.1
    random_state = seed

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        gdb = sklearn.ensemble.GradientBoostingClassifier(n_estimators = n_estimator, learning_rate = learning_rate, max_depth = max_depth, subsample=subsample, random_state = random_state,min_samples_split=min_samples_split,min_samples_leaf = min_samples_leaf,verbose = 1)


        gdb.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = gdb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + gdb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[53]:

### Linear SVM big
def Linear_SVM_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    C = 0.1

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        svc = LinearSVC(C = C, verbose = 1)
        svc.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = svc.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + svc.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[56]:

def Linear_SVM_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.svm import LinearSVC
    C = 0.01

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        svc = LinearSVC(C = C, verbose = 1)
        svc.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = svc.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + svc.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[126]:

### Linear SVM big
def Naive_Bayes_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.naive_bayes import BernoulliNB

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        gnb = BernoulliNB()
        gnb.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + gnb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[129]:

def Naive_Bayes_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.naive_bayes import BernoulliNB

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        gnb = BernoulliNB()
        gnb.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + gnb.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[141]:

def KNN1_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    feature_list = np.array(['num_0','num_0_food','num_0_promo','num_0_trans','trans3','trans4','cate2','food22','cate3','num_na','food19','trans6',
                              'promo9','trans2','promo8','food10','food11','promo12','food16','promo11','trans5','food23','promo6','food18',
                              'food13','cate1','food17','food3','promo2','food14','food21','food9','promo10','food8'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0
    c = 85


    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        X_train_noid = X_train_noid[feature_list]
        X_valid_noid =X_valid_noid[feature_list]
        X_test_noid = X_test_noid[feature_list]
        #gnb = BernoulliNB()
        #gnb.fit(X_train_noid,train_label)
        bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=85),max_samples=0.6, max_features=0.9)
        bagging.fit(X_train_noid,train_label)
        y_valid = bagging.predict(X_valid_noid)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        #y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + bagging.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)
    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[142]:

def KNN1_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    feature_list = np.array(['trans2','promo1','trans1','NA_cate1','NA_cate3'])

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0
    c = 85


    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        #X_train_noid = X_train_noid[feature_list]
        #X_valid_noid =X_valid_noid[feature_list]
        #X_test_noid = X_test_noid[feature_list]
        #gnb = BernoulliNB()
        #gnb.fit(X_train_noid,train_label)
        bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=31),max_samples=0.7, max_features=0.8)
        bagging.fit(X_train_noid,train_label)
        y_valid = bagging.predict(X_valid_noid)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        #y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + bagging.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0

    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[143]:

def KNN2_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    feature_list = np.array(['trans3','trans4','trans6','trans3','cate4','food10','trans1','food11'                          ,'trans5','food3','cate2','promo1'])

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0
    c = 85


    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        #X_train_noid = X_train_noid[feature_list]
        #X_valid_noid =X_valid_noid[feature_list]
        #X_test_noid = X_test_noid[feature_list]
        #gnb = BernoulliNB()
        #gnb.fit(X_train_noid,train_label)
        bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=93),max_samples=0.8, max_features=0.7)
        bagging.fit(X_train_noid,train_label)
        y_valid = bagging.predict(X_valid_noid)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        #y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + bagging.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)
        print (f1_score.f1_eval(y_valid,valid_label))

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[97]:

def Neural_Network_1_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from keras.regularizers import l2, activity_l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop, SGD, adam
    from keras.layers.advanced_activations import PReLU

    model = Sequential()
    #model.add(Dense(1, input_dim=50, activation='sigmoid'))
    np.random.seed(1)
    sgd = SGD(lr=0.01, momentum=0.05, decay=1e-7, nesterov=False)

    model.add(Dense(input_dim = 55, output_dim = 200, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 200, output_dim = 200, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 200, output_dim = 1, init = 'uniform', activation='sigmoid'))
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values.reshape(len(test_result),1)

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NA
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_train_noid = X_train_noid.as_matrix()
        train_label = train_label.as_matrix()
        model.fit(X_train_noid, train_label,nb_epoch=40,batch_size=16)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        X_valid_noid = X_valid_noid.as_matrix()
        y_valid = model.predict_classes(X_valid_noid, batch_size=32)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid.flatten()})
        valid_pred = valid_pred.append(valid_sets)
        X_test_noid = X_test_noid.as_matrix()
        y_pred = y_pred + model.predict_classes(X_test_noid, batch_size=32)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[92]:

def Neural_Network_2_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from keras.regularizers import l2, activity_l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop, SGD, adam
    from keras.layers.advanced_activations import PReLU

    model = Sequential()
    #model.add(Dense(1, input_dim=50, activation='sigmoid'))
    np.random.seed(1)
    sgd = SGD(lr=0.01, momentum=0.05, decay=1e-7, nesterov=False)

    model.add(Dense(input_dim = 55, output_dim = 200, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(Dense(input_dim = 50, output_dim = 200, init = 'uniform', activation='tanh'))
    #model.add(Dropout(0.5))
    #model.add(Dense(input_dim = 200, output_dim = 1, init = 'uniform', activation='sigmoid',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    model.add(Dense(input_dim = 200, output_dim = 1, init = 'uniform', activation='sigmoid'))
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values.reshape(len(test_result),1)

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NA
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_train_noid = X_train_noid.as_matrix()
        train_label = train_label.as_matrix()
        model.fit(X_train_noid, train_label,nb_epoch=30,batch_size=16)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        X_valid_noid = X_valid_noid.as_matrix()
        y_valid = model.predict_classes(X_valid_noid, batch_size=32)

        #         make validation results as a DataFrame
        #print (y_valid.shape)
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid.flatten()})
        valid_pred = valid_pred.append(valid_sets)
        X_test_noid = X_test_noid.as_matrix()
        y_pred = y_pred + model.predict_classes(X_test_noid, batch_size=32)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[ ]:

# xgb tree big data
#def xgb_big_tree(train_list,valid_list,test,booster,seed):
def xgb_big_tree_ensemble(train_list,valid_list,test,seed):
    import xgboost as xgb
    #import f1_score
    #import pandas as pd
    #import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    x_parameters = {"max_depth":6, "objective": "multi:softmax",
                    'silent':1, "eval_metric": "error", "num_class":2,
                    "learning_rate": 0.05, 'silent':1, 
                    "subsample": 0.8,"seed":seed,"min_child_weight": 0.5, "colsample_bytree":0.8,
                    "colsample_bylevel":0.8,"eta":0.4}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    test_result["result"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/(5.0)
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[108]:

def Neural_Network_big_ensemble(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from keras.regularizers import l2, activity_l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop, SGD, adam
    from keras.layers.advanced_activations import PReLU

    model = Sequential()
    #model.add(Dense(1, input_dim=50, activation='sigmoid'))
    np.random.seed(1)
    sgd = SGD(lr=0.01, momentum=0.05, decay=1e-7, nesterov=False)

    model.add(Dense(input_dim = 13, output_dim = 100, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 100, output_dim = 100, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 100, output_dim = 1, init = 'uniform', activation='sigmoid'))
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values.reshape(len(test_result),1)

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NA
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        #pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        #X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        #X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        #X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = test.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_train_noid = X_train_noid.as_matrix()
        train_label = train_label.as_matrix()
        model.fit(X_train_noid, train_label,nb_epoch=40,batch_size=16)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        X_valid_noid = X_valid_noid.as_matrix()
        y_valid = model.predict_classes(X_valid_noid, batch_size=32)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid.flatten()})
        valid_pred = valid_pred.append(valid_sets)
        X_test_noid = X_test_noid.as_matrix()
        y_pred = y_pred + model.predict_classes(X_test_noid, batch_size=32)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)
    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


def Neural_Network_big_ensemble_2(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from keras.regularizers import l2, activity_l2
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.optimizers import RMSprop, SGD, adam
    from keras.layers.advanced_activations import PReLU

    model = Sequential()
    #model.add(Dense(1, input_dim=50, activation='sigmoid'))
    np.random.seed(1)
    sgd = SGD(lr=0.01, momentum=0.05, decay=1e-7, nesterov=False)

    model.add(Dense(input_dim = 68, output_dim = 300, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 300, output_dim = 200, init = 'uniform', activation='tanh', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim = 200, output_dim = 1, init = 'uniform', activation='sigmoid'))
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])


    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values.reshape(len(test_result),1)

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NA
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        #pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        #X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        #X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        #X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = test.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)

        X_train_noid = X_train_noid.as_matrix()
        train_label = train_label.as_matrix()
        model.fit(X_train_noid, train_label,nb_epoch=40,batch_size=16)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)
        X_valid_noid = X_valid_noid.as_matrix()
        y_valid = model.predict_classes(X_valid_noid, batch_size=32)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid.flatten()})
        valid_pred = valid_pred.append(valid_sets)
        X_test_noid = X_test_noid.as_matrix()
        y_pred = y_pred + model.predict_classes(X_test_noid, batch_size=32)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)
    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score



# In[109]:

#xgb tree small data
#def xgb_na(train_list,valid_list,test,booster,seed):
def xgb_na_tree_ensemble(train_list,valid_list,test,seed):
    import xgboost as xgb
    import f1_score
    import pandas as pd
    import numpy as np

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #            'silent':1, "eval_metric": "error", "num_class":2,
    #            "lambda":10, "seed":5, "lambda_bias": 10}
    #if booster == "tree":
    x_parameters = {"max_depth":3, "objective": "multi:softmax", "num_class":2,
                    "learning_rate": 0.05, 'silent':1, "min_child_weight": 0.5, "colsample_bytree":0.8,
                    "colsample_bylevel":0.8,"eta":0.4}

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DTrain = xgb.DMatrix(X_train_noid,label = train_label)

        valid = valid_list[i]
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        DValid = xgb.DMatrix(X_valid_noid)

        X_test_noid = test.drop("Cust_id", axis = 1)
        DTest = xgb.DMatrix(X_test_noid)

        xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = (xgbm.predict(DValid)>0.5)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred += (xgbm.predict(DTest)>0.5)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


# In[ ]:

def Logistic_Regression_big(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import LinearSVC
    C = 0.3

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        #train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)
        from sklearn import linear_model, datasets
        logistic = linear_model.LogisticRegression(C=0.02,penalty='l1', tol=0.03,random_state = seed)
        logistic.fit(X_train_noid,train_label)
        #svc.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = logistic.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + logistic.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score



# In[ ]:

def Logistic_Regression_na(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.svm import LinearSVC
    C = 0.01

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0

    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        #X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        train_label = train["Active_Customer"]
        pca_dict,scaler_dict, X_train, y_train = train_pca_na(train)
        X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        X_valid = test_pca_na(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = X_valid.drop("Cust_id", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        X_test = test_pca_na(scaler_dict,pca_dict,test)
        X_test_noid = X_test.drop("Cust_id", axis = 1)

        from sklearn import linear_model, datasets
        logistic = linear_model.LogisticRegression(C=0.1,penalty='l1', tol=0.03,random_state = seed)
        logistic.fit(X_train_noid,train_label)

        #svc = LinearSVC(C = C, verbose = 1)
        #svc.fit(X_train_noid,train_label)

        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        y_valid = logistic.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + logistic.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score


def KNN2_big_ensemble(train_list,valid_list,test,seed):

    #x_parameters = {"booster":"gblinear","objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "lambda":200, "seed":seed, "lambda_bias": 50}
    #if booster == "tree":
    #x_parameters = {"max_depth":6, "objective": "multi:softmax",
    #                'silent':1, "eval_metric": "error", "num_class":2,
    #                "learning_rate": 0.05, 'silent':1, "min_child_weight": 2,
    #                "subsample": 0.8,"seed":seed}

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    feature_list = np.array(['trans3','trans4','trans6','trans3','cate4','food10','trans1','food11'                          ,'trans5','food3','cate2','promo1'])

    test_result =pd.DataFrame(test["Cust_id"].values,columns=["Cust_id"])
    test_result["y_pred"] = 0
    y_pred = test_result["y_pred"].values

    valid_pred = pd.DataFrame({"Cust_id":["0"],"Active_Customer":[0]})
    valid_pred

    cv_list = []
    cv_score = 0
    c = 85


    for i in range(len(train_list)):
        #     without NAs
        train = train_list[i]
        train_label = train["Active_Customer"]
        X_train_noid = train.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #train_label = train["Active_Customer"]
        #pca_dict,scaler_dict, X_train, y_train = train_pca(train)
        #X_train_noid = X_train.drop("Cust_id", axis = 1)
        #DTrain = xgb.DMatrix(X_train_noid,label = train_label)


        valid = valid_list[i]
        #X_valid = test_pca(scaler_dict,pca_dict,valid)
        valid_label = valid["Active_Customer"]
        X_valid_noid = valid.drop("Cust_id", axis = 1).drop("Active_Customer", axis = 1)
        #DValid = xgb.DMatrix(X_valid_noid)
        #X_test = test_pca(scaler_dict,pca_dict,test)
        X_test_noid = test.drop("Cust_id", axis = 1)
        #X_train_noid = X_train_noid[feature_list]
        #X_valid_noid =X_valid_noid[feature_list]
        #X_test_noid = X_test_noid[feature_list]
        #gnb = BernoulliNB()
        #gnb.fit(X_train_noid,train_label)
        bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=93),max_samples=0.8, max_features=0.7)
        bagging.fit(X_train_noid,train_label)
        y_valid = bagging.predict(X_valid_noid)
        #DTest = xgb.DMatrix(X_test_noid)

        #xgbm = xgb.train(x_parameters, DTrain, num_boost_round=20)

        #y_valid = gnb.predict(X_valid_noid)

        #         make validation results as a DataFrame
        valid_sets = pd.DataFrame({"Cust_id":valid["Cust_id"],"Active_Customer":y_valid})
        valid_pred = valid_pred.append(valid_sets)

        y_pred = y_pred + bagging.predict(X_test_noid)


        cv_list.append(f1_score.f1_eval(y_valid,valid_label))
        cv_score = cv_score + f1_score.f1_eval(y_valid,valid_label)

    cv_score = cv_score/5.0
    print (cv_score)

    test_result["result"] = y_pred
    test_result["Active_Customer"] = 0
    test_result["Active_Customer"][test_result["result"] > 2]=1
    test_result = test_result[["Cust_id","Active_Customer"]]

    valid_pred = valid_pred[1:]

    return test_result, valid_pred, cv_score




# In[ ]:

# main function
import sklearn
import pandas as pd
import numpy as np
from KFoldPCA import *
import f1_score
K = 5

#train_clean = pd.read_csv("train_clean.csv")
#train_list,valid_list = generate_kfold(train_clean,2,K)
#test_clean = pd.read_csv("test_clean.csv")

#test_result, valid_pred, cv_score = KNN1_big(train_list,valid_list,test_clean,5)



#train_clean_na = pd.read_csv("train_clean_na.csv")
#train_list_na,valid_list_na = generate_kfold(train_clean_na,2,K)

#train_list_na,valid_list_na = generate_kfold(train_clean_na,2,K)
#test_clean_na = pd.read_csv("test_clean_na.csv")
#test_result_na, valid_pred_na, cv_score_na =  KNN1_na(train_list_na,valid_list_na,test_clean_na,5)


# In[131]:

#print (cv_score,cv_score_na)


# In[103]:

#test_result.to_csv("big_nn.csv", index = False)


# In[104]:

#test_result_na.to_csv("small_nn.csv", index = False)

