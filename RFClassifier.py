#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


dataset=pd.read_csv("../desktop/adult.csv")


# In[ ]:


#checking null values
dataset.isnull().sum()
dataset.dtypes


# In[ ]:


dataset.head()


# In[ ]:


#removing '?' containing rows
dataset = dataset[(dataset != '?').all(axis=1)]
#label the income objects as 0 and 1
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})


# In[ ]:


#we can reformat marital.status values to single and married
dataset['marital.status']=dataset['marital.status'].map({'Married-civ-spouse':'Married', 'Divorced':'Single', 'Never-married':'Single', 'Separated':'Single', 
'Widowed':'Single', 'Married-spouse-absent':'Married', 'Married-AF-spouse':'Married'})


# In[ ]:


for column in dataset:
    enc=LabelEncoder()
    if dataset.dtypes[column]==np.object:
        dataset[column]=enc.fit_transform(dataset[column])


# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(dataset.corr(),annot=True,fmt='.2f')
plt.show()


# In[ ]:


dataset=dataset.drop(['relationship','education'],axis=1)

dataset=dataset.drop(['occupation','fnlwgt','native.country'],axis=1)

X=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1]
print(X.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,shuffle=False)


# In[ ]:


rf_gini = RandomForestClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=5)
rf_gini.fit(X_train, y_train)
### random forest with Information Gain ###

rf_entropy = RandomForestClassifier(criterion = "entropy", random_state = 100,
 max_depth=5, min_samples_leaf=5)

rf_entropy.fit(X_train, y_train)

y_pred_gini = rf_gini.predict(X_test)
y_pred_en = rf_entropy.predict(X_test)

import pickle

#serializing our model to a file called model.pkl
pickle.dump(dt_clf_gini, open("/model.pkl","wb"))


# In[ ]:





# In[ ]:




