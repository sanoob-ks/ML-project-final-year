#!/usr/bin/env python
# coding: utf-8

# In[1]:
file1=open("my_file.txt","w")
file1.writelines("started \n")
print("status: Starting Proces")
file1.writelines("importing  libraries \n")

print("status: Importing  libraries ")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

import sys
data_location=sys.argv[1]
target_col =sys.argv[2]
cardinality_limit=sys.argv[3]
corr_matrix_threshold=sys.argv[4]
variance_threshold=sys.argv[5]
#1


# In[3]:

file1.writelines("Importing data \n")
print("status: Importing data")
raw_df=pd.read_csv(data_location,encoding="ISO-8859-1")
num_columns=raw_df.shape[1]

# In[4]:


raw_df.fillna(value = 0, inplace = True)


# In[5]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:
#file1.writelines("plotting target feature \n")
#print("status: plotting target feature")

#late_delivery_risk_plot=raw_df.groupby(target_col)[target_col].count()
#late_delivery_risk_plot.plot(kind="bar",title="nothing")
#plt.show()


# In[7]:


raw_df[target_col].value_counts()


# In[8]:

file1.writelines("setting input columns \n")
print("status: Setting input columns ")

input_cols = ["Type",'Days for shipment (scheduled)','Benefit per order','Sales per customer','Category Id','Category Name','Customer City','Customer Country','Customer Segment','Customer State','Customer Street','Department Id','Department Name','Latitude','Longitude','Market', 'Order City','Order Country','Order Item Discount','Order Item Discount Rate','Order Item Product Price','Order Item Profit Ratio','Order Item Quantity','Sales','Order Item Total','Order Profit Per Order','Order Region','Order State','Order Status','Product Card Id','Product Category Id','Product Name','Product Price','Shipping Mode']
#input_cols=raw_df.columns.tolist()
#input_cols


# In[9]:


#feature engineering


# In[10]:


#input_cols=raw_df.columns.tolist()
#input_cols


# In[12]:


#input_cols.remove(target_col)
input_cols


# In[13]:


input_data=raw_df[input_cols]


# In[14]:


corr_data=raw_df[input_cols+[target_col]]


# In[15]:


input_data.info()


# In[16]:

file1.writelines("divide input columns to numerical and categorical \n")
print("status: Categorizing input columns to numerical and categorical ")
#divide input columns to numerical and categorical
numeric_cols =input_data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = input_data.select_dtypes('object').columns.tolist()


# In[17]:


raw_df[categorical_cols].nunique()


# In[18]:


#some columns have small number of unique values. They can be directly one-hot encoded.


# In[19]:

file1.writelines("dividing categorical columns into small and big \n")
print("status: Grouping categorical columns into small and big")
cardinality_limit=5
categorical_small=[]
for i in categorical_cols:
    if raw_df[i].nunique() <= cardinality_limit:
        categorical_small.append(i)
categorical_small        
        


# In[20]:


#others are put into another list, categorical_big
categorical_big=[]
for i in categorical_cols:
    if i not in categorical_small:
        categorical_big.append(i)
categorical_big

file1.writelines("dividing end \n")
print("status: features separated")
# In[21]:


categorical_big


# In[22]:


numeric_cols_data=input_data[numeric_cols]


# In[23]:


df_main=raw_df[input_cols+[target_col]]


# In[24]:


df_main.columns


# In[25]:


#normalization of numerical variables


# In[26]:
file1.writelines("importing MinMaxScaler \n")
print("status: importing MinMaxScaler")
from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler = MinMaxScaler()


# In[28]:


scaler.fit(df_main[numeric_cols])


# In[29]:


df_main[numeric_cols] = scaler.transform(df_main[numeric_cols])


# In[30]:


df_main[numeric_cols].describe()


# In[31]:


# encoding categorical variables


# In[32]:
file1.writelines("one-hot encoding start(for small number of unique values) \n")
print("status: one-hot encoding started(for small cardinality features)")
from sklearn.preprocessing import OneHotEncoder


# In[33]:


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[34]:


encoder.fit(df_main[categorical_small])


# In[35]:


encoded_cols = list(encoder.get_feature_names(categorical_small))


# In[36]:


df_main[encoded_cols] = encoder.transform(df_main[categorical_small])


# In[37]:


encoded_cols

file1.writelines("one-hot encoding end \n")
print("status: one-hot encoding completed ")
# In[38]:
file1.writelines("target_guided_encoding started \n")
print("status: target_guided_encoding started")
from sunbird.categorical_encoding import target_guided_encoding


# In[39]:


for i in categorical_big:
    target_guided_encoding(df_main, i,target_col)

file1.writelines("target_guided_encoding completed \n")
print("status: target_guided_encoding completed ")

# In[40]:


#raw_df[categorical_big].head(10)


# In[41]:


categorical_big


# In[42]:


df_main.columns


# In[43]:


df_main.info()


# In[44]:


corr_1=corr_data.corr()


# In[45]:



encoded_cols=encoded_cols+categorical_big
encoded_cols


# In[46]:


#feature selection


# In[47]:
file1.writelines("chi2 test start \n")
print("status: chi2 test started ")
from sklearn.feature_selection import chi2


# In[48]:


X = df_main[encoded_cols]
y = df_main[target_col]


# In[49]:


chi_scores = chi2(X,y)


# In[50]:


chi_scores


# In[51]:


p_values =p_values_1= pd.Series(chi_scores[1],index = X.columns)


# In[52]:


categorical_selected=[]
for i in range(len(p_values_1)):
    if p_values_1[i]<0.1:
        categorical_selected.append(encoded_cols[i])
categorical_selected


# In[53]:


p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()


# In[54]:


X = df_main[categorical_big]
y = df_main[target_col]
chi_scores = chi2(X,y)
p_values =p_values_1= pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar(xlabel='Encoded Columns',ylabel='p - value',title='Chi-Square Test for Target Guided Encoded')


# In[55]:


X = df_main[encoded_cols+categorical_big]
y = df_main[target_col]
chi_scores = chi2(X,y)
p_values =p_values_1= pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar(xlabel='Encoded Columns',ylabel='p - value',title='Chi-Square Test for Target Guided Encoded')

file1.writelines("chi2 test end \n")
print("status:chi2 test completed ")
# In[56]:


#feature selection for numerical variables.


# In[57]:
file1.writelines("creating correlation matrix \n")
print("status: creating correlation matrix ")
plt.figure(figsize=(10,10))
sns.heatmap(corr_1, cbar=True, square=True, fmt='.2f', annot=True,annot_kws={'size':8}, cmap='Blues')


# In[58]:


upper_tri =corr_1.where(np.triu(np.ones(corr_1.shape),k=1).astype(np.bool))


# In[59]:


corr_matrix_threshold=0.95


# In[60]:


correlated = [column for column in upper_tri.columns if any(upper_tri[column] > corr_matrix_threshold)]
print(correlated)


# In[ ]:





# In[61]:


#df1 = df.drop(df.columns[to_drop], axis=1)


# In[62]:


#order item product price 
#Order profit per order
#Order item total
#Sales per customer
#Order item product price
#order item cardpod id


# In[63]:


numeric_cols


# In[64]:

file1.writelines("removing correlated columns \n")
print("status: removing correlated columns")
numeric_after_corr=[]
for i in numeric_cols:
    if i not in correlated:
        numeric_after_corr.append(i)


# In[65]:


numeric_after_corr


# In[ ]:





# In[66]:


#variance threshold


# In[67]:


variance_threshold=0.01


# In[68]:

file1.writelines("VarianceThreshold start \n")
print("status: VarianceThreshold started ")
from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(threshold=variance_threshold)


# In[69]:


df_var=df_main[numeric_after_corr]


# In[70]:


df_var.shape


# In[71]:


transformed = vt.fit_transform(df_var)


# In[72]:


_ = vt.fit(df_var)

mask = vt.get_support()


# In[73]:


df_var_reduced = df_var.loc[:, mask]


# In[74]:


df_var_reduced.shape


# In[75]:


i=df_var_reduced.columns


# In[76]:


df_var_reduced.columns


# In[77]:


numeric_cols_selected=[]
for x in i:
    numeric_cols_selected.append(x)


# In[ ]:





# In[78]:


numeric_cols_selected


# In[79]:


input_cols_final=categorical_selected
for i in numeric_cols_selected:
    input_cols_final.append(i)


# In[80]:


input_cols_final


# In[81]:


df_main.head()


# In[82]:


#model building


# In[83]:


recall_list=[]


# In[84]:

file1.writelines(" train test split for catboost  start \n")
print("status: train test split for catboost  started")
from sklearn.model_selection import train_test_split


# In[85]:



train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)





# In[ ]:





# In[86]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[87]:


test_inputs=test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

file1.writelines(" train test split for catboost end \n")
print("status: train test split for catboost ended")
# In[88]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# In[89]:

if num_columns<30:
    file1.writelines("catboost model building start \n")
    print("status: catboost model building started")
    import catboost
    from catboost import CatBoostClassifier
    model=CatBoostClassifier()
    model.fit(train_inputs,train_targets,cat_features=categorical_cols)
    test_preds_catboost = model.predict(test_inputs)

    recall=recall_score(test_targets, test_preds_catboost)
    recall_list.append(recall)
    file1.writelines("catboost model building end \n")
    print("status: catboost model building end")
else:
    recall_list.append(0)
    print("status: catboost model building skipped")
# In[91]:



train_val_df, test_df = train_test_split(df_main, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[93]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# In[94]:


train_df.head()


# In[ ]:





# In[95]:


train_inputs = train_df[input_cols_final].copy()
train_targets = train_df[target_col].copy()


# In[96]:


val_inputs = val_df[input_cols_final].copy()
val_targets = val_df[target_col].copy()


# In[97]:


test_inputs = test_df[input_cols_final].copy()
test_targets = test_df[target_col].copy()


# In[99]:


train_inputs.head()


# In[ ]:





# In[100]:

file1.writelines("Logistic Regression model building start \n")
print("status: Logistic Regression model building started")
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(train_inputs, train_targets)


# In[101]:


X_train = train_inputs[input_cols_final]
X_val = val_inputs[input_cols_final]
X_test = test_inputs[input_cols_final]


# In[102]:


train_preds_logistic = model.predict(X_train)


# In[103]:


train_probs_logistic = model.predict_proba(X_train)


# In[104]:


test_preds_logistic = model.predict(X_test)


# In[105]:


recall=recall_score(test_targets, test_preds_logistic)
recall_list.append(recall)

file1.writelines("Logistic Regression model building end \n")
print("status: Logistic Regression model building completed")
# In[106]:


#print('train_inputs:', train_inputs.shape)
#print('train_targets:', train_targets.shape)
#print('val_inputs:', val_inputs.shape)
#print('val_targets:', val_targets.shape)
#print('test_inputs:', test_inputs.shape)
#print('test_targets:', test_targets.shape)


# In[107]:


#test_preds = model.predict(X_test)


# In[108]:


#accuracy_score(test_targets, test_preds)


# In[109]:
file1.writelines("GaussianNB model building strat \n")
print("status: GaussianNB model building strated")

#from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(train_inputs, train_targets)
train_preds_gaussian = model.predict(X_train)
train_probs_gaussian = model.predict_proba(X_train)
test_preds_gaussian = model.predict(X_test)
recall=recall_score(test_targets, test_preds_gaussian)
recall_list.append(recall)

file1.writelines("GaussianNB model building end \n")
print("status: GaussianNB model building completed")
# In[110]:
file1.writelines("RandomForestClassifier model building strat \n")
print("status: RandomForestClassifier model building strated")
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)#doubt
model.fit(train_inputs, train_targets)
train_preds_random = model.predict(X_train)
train_probs_random= model.predict_proba(X_train)
test_preds_random = model.predict(X_test)
recall=recall_score(test_targets, test_preds_random)
recall_list.append(recall)

file1.writelines("RandomForestClassifier model building end \n")
print("status: RandomForestClassifier model building completed")
# In[111]:

file1.writelines("xgboost model building strat \n")
print("status: xgboost model building strated")
import xgboost as xgb
from xgboost import XGBClassifier
model=xgb.XGBClassifier()
model.fit(train_inputs, train_targets)
train_preds_xgb = model.predict(X_train)
train_probs_xgb = model.predict_proba(X_train)
test_preds_xgb = model.predict(X_test)
recall=recall_score(test_targets, test_preds_xgb)
recall_list.append(recall)

file1.writelines("xgboost model building end \n")
print("status: xgboost model building completed")
# In[112]:
file1.writelines("lightgbm model building strat \n")
print("status: lightgbm model building strated")

import lightgbm as lgb
from lightgbm import LGBMClassifier
model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(train_inputs, train_targets)
train_preds_lgb = model.predict(X_train)
train_probs_lgb = model.predict_proba(X_train)
test_preds_lgb = model.predict(X_test)
recall=recall_score(test_targets, test_preds_lgb)
recall_list.append(recall)
file1.writelines("lightgbm model building start \n")
print("status: lightgbm model building started")
# In[ ]:





# In[113]:


recall_list


# In[114]:


index=recall_list.index(max(recall_list))


# In[115]:

file1.writelines("printing all score for model with high recall_score  \n")
print("status: finding best model and getting results")
if index==0:
    
    recall_score=recall_score(test_targets, test_preds_catboost)
    accuracy_score=accuracy_score(test_targets, test_preds_catboost)
    f1_score=f1_score(test_targets, test_preds_catboost)
    precision_score=precision_score(test_targets, test_preds_catboost)
    roc_auc_score=roc_auc_score(test_targets, test_preds_catboost)

    print("result1","Catboost")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
   
    rf=CatBoostClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets) 

elif index==1:
    recall_score=recall_score(test_targets, test_preds_logistic)
    accuracy_score=accuracy_score(test_targets, test_preds_logistic)
    f1_score=f1_score(test_targets, test_preds_logistic)
    precision_score=precision_score(test_targets, test_preds_logistic)
    roc_auc_score=roc_auc_score(test_targets, test_preds_logistic)

    print("result1","Logistic Regression")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=LogisticRegression(solver='liblinear',multi_class='ovr')
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
    
elif index==2:
    recall_score=recall_score(test_targets, test_preds_gaussian)
    accuracy_score=accuracy_score(test_targets, test_preds_gaussian)
    f1_score=f1_score(test_targets, test_preds_gaussian)
    precision_score=precision_score(test_targets, test_preds_gaussian)
    roc_auc_score=roc_auc_score(test_targets, test_preds_gaussian)

    print("result1","Gaussin NB")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=GaussianNB()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==3:
    recall_score=recall_score(test_targets, test_preds_random)
    accuracy_score=accuracy_score(test_targets, test_preds_random)
    f1_score=f1_score(test_targets, test_preds_random)
    precision_score=precision_score(test_targets, test_preds_random)
    roc_auc_score=roc_auc_score(test_targets, test_preds_random)

    print("result1","Random Forest")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=RandomForestClassifier(n_estimators=50)
    rf.fit(test_inputs,test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==4:
    recall_score=recall_score(test_targets, test_preds_xgb)
    accuracy_score=accuracy_score(test_targets, test_preds_xgb)
    f1_score=f1_score(test_targets, test_preds_xgb)
    precision_score=precision_score(test_targets, test_preds_xgb)
    roc_auc_score=roc_auc_score(test_targets, test_preds_xgb)

    print("result1","XG Boost")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=XGBClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
elif index==5:
    recall_score=recall_score(test_targets, test_preds_lgb)
    accuracy_score=accuracy_score(test_targets, test_preds_lgb)
    f1_score=f1_score(test_targets, test_preds_lgb)
    precision_score=precision_score(test_targets, test_preds_lgb)
    roc_auc_score=roc_auc_score(test_targets, test_preds_lgb)

    print("result1","Lightgbm")
    print("result2",recall_score)
    print("result3",accuracy_score)
    print("result4",f1_score)
    print("result5",precision_score)
    print("result6",roc_auc_score)
    
    rf=LGBMClassifier()
    rf.fit(test_inputs, test_targets)
    metrics.plot_roc_curve(rf,X_test,test_targets)
    
  


# In[ ]:


file1.write("All programme runned")
print("status: Model Building Completed")
file1.close()



# In[ ]:





