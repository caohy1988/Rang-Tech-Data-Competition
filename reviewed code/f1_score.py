
# coding: utf-8

# In[30]:

def f1_eval(y_true,y_pred):
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, y_pred, average='binary') 
    return f1


# In[ ]:



