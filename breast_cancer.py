
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


from sklearn.datasets import load_breast_cancer
Breast_cancer = load_breast_cancer()

Breast_cancer

Breast_cancer_df = pd.DataFrame(np.c_[Breast_cancer['data'], Breast_cancer['target']], columns = np.append(Breast_cancer['feature_names'], ['target']))

Breast_cancer_df.head()

Breast_cancer_df.shape
Breast_cancer_df.columns

sns.pairplot(Breast_cancer_df, hue = 'target',palette='gnuplot2', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )

Breast_cancer_df['target'].value_counts()
sns.countplot(Breast_cancer_df['target'], label = "Count")

Train_data = Breast_cancer_df.drop(['target'], axis = 1) 
Train_data.head()

Target_data = Breast_cancer_df['target']
Target_data.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Train_data, Target_data, test_size = 0.2, random_state = 20)

print ('Number of training samples input is', X_train.shape)
print ('Number of testing samples input is', X_test.shape)
print ('Number of training samples output is', y_train.shape)
print ('Number of testing samples output is', y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

from sklearn.svm import SVC
svc_model = SVC()

svc_model.fit(X_train_scaled, y_train)

Prediction = svc_model.predict(X_test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

cm = np.array(confusion_matrix(y_test, Prediction, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion



print(classification_report(y_test,Prediction))

import pickle

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(svc_model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

Prediction = pickle_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, Prediction)

cm = np.array(confusion_matrix(y_test, Prediction, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion




