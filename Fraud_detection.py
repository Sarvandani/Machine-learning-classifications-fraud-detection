#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sarvandani
##The data of this work can be found in the following link:

##https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn import metrics



data= pd.read_csv('fraud_oracle.csv')
### data analysis
print(data.shape)
print(data.columns)
data.info()
data.hist(bins=10, figsize=(25, 20))
plt.show()
data.drop_duplicates()
#sns.barplot(x=data['FraudFound_P'].value_counts().index,y=data['FraudFound_P'].value_counts()).set(title='No Fraud vs Fraud totals')
Features=['Month','WeekOfMonth','DayOfWeek','Make','AccidentArea','DayOfWeekClaimed','MonthClaimed','WeekOfMonthClaimed','Sex','MaritalStatus','Age','Fault','PolicyType','VehicleCategory','VehiclePrice','Deductible','DriverRating','Days_Policy_Accident','Days_Policy_Claim','PastNumberOfClaims','AgeOfVehicle','AgeOfPolicyHolder','PoliceReportFiled','WitnessPresent','AgentType','NumberOfSuppliments','AddressChange_Claim','NumberOfCars','Year','BasePolicy']
correlations = data.corr()
## we check correlation with Fraudfound_p and other features, closer to 1 is high
hi_corr=['BasePolicy','VehicleCategory','Fault','VehiclePrice','PolicyType']

###############@
fraud_data = data[data.FraudFound_P==1]

## plot variations between high-correlation and fraud
def get_viz(column):
    fig,(ax1)=plt.subplots(1,1,figsize=(9,4))
    fig.suptitle('high-correlation features and fraud')
    sns.barplot(x=fraud_data[column].value_counts().index,y=fraud_data[column].value_counts(),ax=ax1)
    
def multi_viz(list):
    mv_list=[]
    for x in list:
        if x not in mv_list:
            get_viz(x)
multi_viz(hi_corr)

##Encoding: transforming string to integer data
def transform(column):
    encode=LabelEncoder().fit_transform(data[column])
    data.drop(column,axis=1,inplace=True)
    data[column]=encode
def multi_transform(list):
    mt_list=[]
    for x in list:
        if x not in mt_list:
            transform(x)
multi_transform(Features)

#splitting data into features and lables
y=data.FraudFound_P
X=data[Features]
train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=42)

##############
## cheking if data is balanced or not
print(f"Training target statistics: {Counter(train_y)}")
print(f"Testing target statistics: {Counter(test_y)}")
train_X.shape, test_X.shape, train_y.shape, test_y.shape
##############################
##balancing datasets
from sklearn.utils import class_weight
class_weights = dict(zip(np.unique(train_y), class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(train_y), y = train_y)))
print(class_weights)

###########modeling methods
model1 = DecisionTreeClassifier(class_weight=class_weights)
model1.fit(train_X, train_y)
predictions1=model1.predict(test_X)
model1_score = {}


#####
model2 = RandomForestRegressor(random_state=42)
model2.fit(train_X, train_y)
predictions2=model2.predict(test_X)
#######
model3 = LogisticRegression(random_state=42)
model3.fit(train_X, train_y)
predictions3=model3.predict(test_X)

###########
model4 = XGBClassifier(random_state=42)
model4.fit(train_X, train_y)
predictions4=model4.predict(test_X)

                      
######################@

###########################@
#checking accuracy
#model 1
print('MAE:',mean_absolute_error(test_y,predictions1))
print('Max Error:',max_error(test_y,predictions1))
print('metrics.accuracy_scor:',metrics.accuracy_score(test_y, predictions1))
#model 2
print('MAE:',mean_absolute_error(test_y,predictions2))
print('Max Error:',max_error(test_y,predictions2))
#print('metrics.accuracy_scor:',metrics.accuracy_score(test_y, predictions2))
#model3
print('MAE:',mean_absolute_error(test_y,predictions3))
print('Max Error:',max_error(test_y,predictions3))
print('metrics.accuracy_scor:',metrics.accuracy_score(test_y, predictions3))
#####
#model 4
print('MAE:',mean_absolute_error(test_y,predictions4))
print('Max Error:',max_error(test_y,predictions4))
print('metrics.accuracy_scor:',metrics.accuracy_score(test_y, predictions4))

df = pd.DataFrame({'ACCURACY':['DecisionTreeClassifier', 'RandomForestRegressor', 'LogisticRegression', 'XGBClassifier'], 'MAE':[mean_absolute_error(test_y,predictions1), mean_absolute_error(test_y,predictions2), mean_absolute_error(test_y,predictions3), mean_absolute_error(test_y,predictions4)]})
ax = df.plot.bar(x='ACCURACY', y='MAE', color=['cyan'])

df = pd.DataFrame({'ACCURACY':['DecisionTreeClassifier', 'RandomForestRegressor', 'LogisticRegression', 'XGBClassifier'], 'MAXIMUM_ERROR':[max_error(test_y,predictions1), max_error(test_y,predictions2), max_error(test_y,predictions3), max_error(test_y,predictions4)]})
ax = df.plot.bar(x='ACCURACY', y='MAXIMUM_ERROR', color=['green'])

df = pd.DataFrame({'ACCURACY':['DecisionTreeClassifier', 'LogisticRegression', 'XGBClassifier'], 'metrics.accuracy_score':[metrics.accuracy_score(test_y, predictions1), metrics.accuracy_score(test_y, predictions3), metrics.accuracy_score(test_y, predictions4)]})
ax = df.plot.bar(x='ACCURACY', y='metrics.accuracy_score', color=['red'])


