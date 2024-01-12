#!/usr/bin/env python
# coding: utf-8

# In[302]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[303]:


train_data = pd.read_csv('E:/BACKUP/users-admin files/Downloads/Training Matches IPL 2008-2020.csv')
train_data.head()


# In[304]:


train_data.isnull().sum()


# In[305]:


train_data['city'].fillna('Abu Dhabi',inplace=True)
train_data['winner'].fillna('Draw', inplace = True)


# In[306]:


#Both Rising Pue Supergiant and Rising Pune Supergiants represents same team similarly Delhi Capitals and Delhi Daredevils,
#Deccan Chargers and Sunrisers Hyderabad
train_data.replace("Rising Pune Supergiant","Rising Pune Supergiants", inplace=True)
train_data.replace('Deccan Chargers', 'Sunrisers Hyderabad', inplace=True)
train_data.replace('Delhi Daredevils', 'Delhi Capitals', inplace=True)


# In[307]:


plt.subplots(figsize = (15,5))
sns.countplot(x = 'season' , data = train_data, palette='dark')
plt.title('Total number of matches played in each season')
plt.show()


# In[308]:


plt.subplots(figsize=(15,10))
ax = train_data['winner'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette("husl", 9))
ax.set_xlabel('count')
ax.set_ylabel('team')
plt.show()


# In[309]:


#Extracting the records where a team won batting first
batting_first=train_data[train_data['win_by_runs']!=0]
#Making a pie chart
plt.figure(figsize=(7,7))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[310]:


#extracting those records where a team has won after batting second
batting_second=train_data[train_data['win_by_wickets']!=0]
#Making a pie chart for distribution of most wins after batting second
plt.figure(figsize=(7,7))
plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[311]:


sns.countplot(train_data['winner'])
plt.xticks(rotation = 90)


# In[312]:


train_data = train_data.drop(columns=['date','umpire1','umpire2','toss_winner','win_by_runs','win_by_wickets','toss_decision','toss_decision','result','dl_applied','player_of_match'])
print(train_data)
sns.heatmap(train_data.isnull())
plt.show()


# In[313]:


train_data.replace({"Mumbai Indians":"MI", "Delhi Capitals":"DC", 
               "Sunrisers Hyderabad":"SRH", "Rajasthan Royals":"RR", 
               "Kolkata Knight Riders":"KKR", "Kings XI Punjab":"KXIP", 
               "Chennai Super Kings":"CSK", "Royal Challengers Bangalore":"RCB",
              "Kochi Tuskers Kerala":"KTK", "Rising Pune Supergiants":"RPS",
              "Gujarat Lions":"GL", "Pune Warriors":"PW"}, inplace=True)


encode = {'team1': {'KKR':1,'CSK':2,'RR':3,'MI':4,'SRH':5,'KXIP':6,'RCB':7,'DC':8,'KTK':9,'RPS':10,'GL':11,'PW':12},
         'team2': {'KKR':1,'CSK':2,'RR':3,'MI':4,'SRH':5,'KXIP':6,'RCB':7,'DC':8,'KTK':9,'RPS':10,'GL':11,'PW':12},
          'winner': {'KKR':1,'CSK':2,'RR':3,'MI':4,'SRH':5,'KXIP':6,'RCB':7,'DC':8,'KTK':9,'RPS':10,'GL':11,'PW':12,'Draw':13}}
train_data.replace(encode, inplace=True)
train_data.head(5)


# In[314]:


dicVal = encode['winner']
train = train_data[['team1','team2','city','winner']]
train.head(6)


# In[315]:


df = pd.DataFrame(train)
var_mod = ['city']
le = preprocessing.LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


# In[316]:


X = df[['team1', 'team2','city']]
y = df[['winner']]
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)


# In[317]:


linear_model= LinearRegression()
linear_model.fit(X,y)
print("Linear Regression accuracy: ",(linear_model.score(X,y))*100)
logistic_model = LogisticRegression()
logistic_model.fit(X,y)
print("Logistic Regression accuracy: ",(logistic_model.score(X,y))*100)
Random_model = RandomForestClassifier()
Random_model.fit(X,y)
print("Random Forest accuracy: ", (Random_model.score(X,y))*100)
knn_model = KNeighborsClassifier()
knn_model.fit(X,y)
print("KNeighbor Classifier accuracy", (knn_model.score(X,y))*100)


# In[318]:


#test_data = pd.read_csv('E:/BACKUP/users-admin files/Downloads/Testset Matches IPL 2020.csv')
test_data = pd.read_csv('E:/BACKUP/users-admin files/OneDrive/Desktop/Testset Matches IPL 2021.csv')

encode = {'team1': {'KKR':1,'CSK':2,'RR':3,'MI':4,'SRH':5,'KXIP':6,'RCB':7,'DC':8,'KTK':9,'RPS':10,'GL':11,'PW':12},
         'team2': {'KKR':1,'CSK':2,'RR':3,'MI':4,'SRH':5,'KXIP':6,'RCB':7,'DC':8,'KTK':9,'RPS':10,'GL':11,'PW':12}}
test_data.replace(encode,inplace=True)
var_mod = ['city']
le = preprocessing.LabelEncoder()
for i in var_mod:
    test_data[i] = le.fit_transform(test_data[i].astype(str))


# In[319]:


test_X = test_data[['team1','team2','city']]
test_X = sc.fit_transform(test_X)
y_predict = Random_model.predict(test_X)
newlist = list()
for i in y_predict:
    newlist.append(list(dicVal.keys())[list(dicVal.values()).index(i)]) 
test_data['winner'] = newlist

#Decoding Team Names
for i in range(27):
    test_data['team1'][i]=(list(dicVal.keys())[list(dicVal.values()).index(test_data['team1'][i])]) 
    test_data['team2'][i]=(list(dicVal.keys())[list(dicVal.values()).index(test_data['team2'][i])]) 
test_data.head()
