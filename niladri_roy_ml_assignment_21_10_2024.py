#!/usr/bin/env python
# coding: utf-8

# ## step 1 - import the imp lib

# In[1]:


# data handling and manipulation 

import numpy as np 
import pandas as pd 

# visualization

import matplotlib.pyplot as plt
import seaborn as sns 

#model building 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#model evaluation 

from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,accuracy_score



#Miscellaneous

import warnings
warnings.filterwarnings("ignore") # to ignore warning 


# # step-2 load dataset

# In[2]:


file_path= "Fin_Tech_Data.csv"

df=pd.read_csv(file_path)

# display frst 10 row of the datasets

df.tail(10)


# # step 3 :-data preprocessing 

# In[3]:


#shape of the dataset
df.shape
# data set have 50000 row and 11 columns


# In[4]:


# check for missing value
df.isnull().sum()
# dataset has 18926 missing data in column enrolled_date
# it signify that the user which didnt enrolled 


# In[5]:


# check the data type 
df.info()
# as check that first open and hours few columns are in object format need to change the dtype 


# In[6]:


#converting hour dtype from object to int64
df["hour"]=df["hour"].str.split(":").str[0]
df["hour"]=pd.to_numeric(df["hour"]).astype("int64")
df.head(10)


# In[7]:


df["first_open"]=pd.to_datetime(df["first_open"],errors="coerce")
df.tail(10)


# In[8]:


#check no of row cant be convert 
df["first_open"].isnull().sum()
# even changing the dtype the frst open columns has more 50% data missing so remove the missing row is not
#possible as more than 50% will be remove so for the time being negleting this column in feture selection 


# # step 4 visualize relationship between features 
# 

# In[9]:


# univariate analysis
plt.figure(figsize=(8,6))
sns.histplot(data=df[df["enrolled"]==1]["age"],kde=True,bins=30)
plt.title("Age distribution")
plt.ylabel("frequency")
plt.show()
#as we can data in age is left skew age bwt 20-40 the peak age grp which have enrolled 


# In[10]:


# countplot according to dayofweek subcribed 
plt.figure(figsize=(6,5))
sns.countplot(x="dayofweek",data=df[df["enrolled"]==1],palette="viridis")
plt.title("Enrollment distrubution day wise ")
plt.ylabel("no of enrolled")
plt.show()
#0-6 represented the 7 days in a week where 0 is monday and 6 is sunday
# according to graph monday and sat are the highest enrolled days


# In[11]:


# countplot of enrolled vs not enrolled 
plt.figure(figsize=(6,6))
ax=sns.countplot(x="enrolled",data=df,palette="muted")
# Add the count labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), 
                textcoords='offset points')
plt.xlabel("enrolled status")
plt.title("count pf enrolled (1) and not enrolled (0) users")
plt.ylabel("count")
plt.xticks([0,1],["Not Enrolled(0)","Enrolled (1)"])
plt.show()
# here signify the data in not balance class 1 is higher than class o 


# # bivariate analysis

# In[12]:


# plotting "age" vs " enrolled"
plt.figure(figsize=(6,5))
sns.boxenplot(x=df["enrolled"],y=df["age"],data=df, palette="muted")


# In[13]:


#boxplot of numscreens and enrolled 
plt.figure(figsize=(6,5))
sns.boxenplot(x="enrolled",y="numscreens",data=df,palette="muted")
plt.title("")


# In[16]:


# correlation matrix
plt.figure(figsize=(6,5))
sns.heatmap(x.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
plt.title("correlation Matrix")
plt.show()


# # feature selection  or variable 

# In[15]:


#exclude "first_open" from the feature set 
x=df.drop(columns=["first_open","enrolled_date","enrolled"])  # indepedent variable
y=df["enrolled"] # target variable 


# # step-5 data splitting 

# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[18]:


#check the shape of the dataset 
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# # model selection 

# In[19]:


# model1 logistic regression 
from sklearn.metrics import classification_report
#initialize models
logistic_model=LogisticRegression(random_state=42)


# training the model 
logistic_model.fit(x_train,y_train)


#predictions
logistic_pred=logistic_model.predict(x_test)

#evaluation 
classification_report(y_test,logistic_pred, target_names=["Not Enrolled","Enrolled"])


# In[20]:


#model 2 random forest 
#initialize models
rf_model=RandomForestClassifier(random_state=42)


# training the model 
rf_model.fit(x_train,y_train)


#predictions
rf_model_pred=rf_model.predict(x_test)

#evaluation 
classification_report(y_test,rf_model_pred,target_names=["Not Enrolled","Enrolled"])


# In[21]:


# model 3 (gradient boosting)xgboost 
#initialize models
xgb_model=XGBClassifier(random_state=42)


# training the model 
xgb_model.fit(x_train,y_train)


#predictions
xgb_model_pred=xgb_model.predict(x_test)

#evaluation 
classification_report(y_test,xgb_model_pred,target_names=["Not Enrolled","Enrolled"])


# - conculsion are doing are three model more or less we are get same ranhe of values among all xgboosting model has highest accuracy, precision, recall, f1-score but marginaly not satisfing the due to imbalnce datadet so we are try with balcing the data and doing xgboost again 

# In[22]:


#balancing by class weight in random forest 
#model 2 random forest 
#initialize models
rf_model1=RandomForestClassifier(random_state=42,class_weight="balanced")


# training the model 
rf_model1.fit(x_train,y_train)


#predictions
rf_model_pred1=rf_model1.predict(x_test)

#evaluation 
classification_report(y_test,rf_model_pred1,target_names=["Not Enrolled","Enrolled"])


# In[23]:


# model1 logistic regression 
from sklearn.metrics import classification_report
#initialize models
logistic_model1=LogisticRegression(random_state=42,class_weight="balanced")


# training the model 
logistic_model1.fit(x_train,y_train)


#predictions
logistic_pred1=logistic_model1.predict(x_test)

#evaluation 
classification_report(y_test,logistic_pred1, target_names=["Not Enrolled","Enrolled"])


# In[24]:


# balancing by smote 
get_ipython().system('pip install imbalanced-learn')


# In[25]:


# import smote fromimblearn 
from imblearn.over_sampling import SMOTE


# In[26]:


# create am instance of smote 
smote=SMOTE(random_state=42)

#fit and resample the training data
x_rsmple,yrsmple=smote.fit_resample(x_train,y_train)


#check the shape of the train data sets 
print(x_rsmple.shape)
print(yrsmple.shape)


# In[27]:


# model1 using SMOTE logistic regression 
from sklearn.metrics import classification_report
#initialize models
logistic_model2=LogisticRegression(random_state=42)


# training the model 
logistic_model2.fit(x_rsmple,yrsmple)


#predictions
logistic_pred2=logistic_model2.predict(x_test)

#evaluation 
classification_report(y_test,logistic_pred2, target_names=["Not Enrolled","Enrolled"])


# In[28]:


#balancing by SMOTE in random forest 
#model 2 random forest 
#initialize models
rf_model2=RandomForestClassifier(random_state=42)


# training the model 
rf_model2.fit(x_rsmple,yrsmple)


#predictions
rf_model_pred2=rf_model2.predict(x_test)

#evaluation 
classification_report(y_test,rf_model_pred2,target_names=["Not Enrolled","Enrolled"])


# - after trying balancing using both smote and class_weight method logistic regireesion model predict more well and in smote balancing the logistic regiression has the highest accurancy , f1 score so considering the best model 

# In[29]:


from sklearn.metrics import roc_auc_score, roc_curve

# Predict probabilities on the test set
y_proba = logistic_model2.predict_proba(x_test)[:, 1]  # Use the probability of the positive class (class 1)

# Calculate AUC-ROC
auc_score = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score: {auc_score:.2f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[30]:


# Assuming logistic_model is your final model
feature_importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': logistic_model2.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance:\n", feature_importance)


# # Recommendations
# - Optimize Promotions by Day: Consider focusing marketing campaigns on the days when users are more likely to enroll.
# - Encourage Exploration: Since more screen visits predict higher enrollment, enhance the user journey to promote exploring various screens and features within the app.
# - Leverage Premium Features: Highlight the benefits of premium features, especially to those who already engage with them.
# - This feature importance analysis can guide the company’s marketing and user engagement strategies, making efforts more effective and targeted based on data-driven insights. Let me know if you'd like further recommendations on using these insights in practical applications!
# 
# 
# 
# 
# 
# 
# 

# # users to target 

# In[32]:


# Use the trained model to predict probabilities for the positive class (enrolled = 1)
enrollment_probs = logistic_model2.predict_proba(x_test)[:, 1]
x_test['Enrollment_Probability'] = enrollment_probs


# In[33]:


# Create a DataFrame with User IDs and probabilities
ranked_users = x_test.copy()
ranked_users['User_ID'] = y_test.index  # Adjust if User ID is different
ranked_users = ranked_users[['User_ID', 'Enrollment_Probability']]

# Sort users by probability in descending order
ranked_users = ranked_users.sort_values(by='Enrollment_Probability', ascending=False)
print("Top users by enrollment likelihood:\n", ranked_users.head(10))


# In[ ]:


# this are list of user need to target with feature above mentioned

