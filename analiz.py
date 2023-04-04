# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 01:13:26 2023

@author: Batuhan
"""


import pandas as pd                     
#import numpy as np                    
#import seaborn as sns                 
#import matplotlib.pyplot as plt       
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


import warnings                     
warnings.filterwarnings("ignore")

tahmin =pd.read_csv('train_kredi_tahmini.csv')

print(tahmin)
for each in tahmin.columns:
    mode=tahmin[each].mode()
    tahmin[each].fillna(mode[0],inplace=True)
    
    
tahmin['Education'].replace({"Graduate": 1, "Not Graduate": 0}, inplace=True) 
tahmin['Gender'].replace({"Male":0,"Female":1},inplace = True)
tahmin['Married'].replace({"Yes":1,"No":0},inplace = True)
tahmin['Self_Employed'].replace({"Yes":1,"No":0},inplace = True)
tahmin['Dependents'].replace({"3+":4},inplace = True)
tahmin['Loan_Status'].replace({"Y":1,"N":0},inplace = True)


tahmin = tahmin.drop('Loan_ID', axis=1)
x=tahmin.drop(["Loan_Status"],axis=1)
y=pd.DataFrame(tahmin["Loan_Status"])



tahmin = pd.get_dummies(tahmin, columns=['Property_Area']) 

print(x) 
print(y) 
# Sonuç
print(tahmin)

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(tahmin)
scalerDegerler= scaler.transform(tahmin)


one_hot=OneHotEncoder(handle_unknown="ignore")
one_hot_cinsiyet=one_hot.fit_transform(tahmin["Gender"].values.reshape(-1,1)).toarray()

print(one_hot.categories_)
print(one_hot_cinsiyet)
one_hot_df=pd.DataFrame(one_hot_cinsiyet[:,:2],columns=["Kadın","Erkek"])

df1=tahmin.join(one_hot_df)
df1.drop(["Gender"],axis=1,inplace=True)

x=df1.drop(["Loan_Status"],axis=1)

y=pd.DataFrame(df1["Loan_Status"])




from sklearn.model_selection import train_test_split # istersek train setinden test seti ayırabiliriz örnek


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print(X_train.shape)
print(X_test.shape)





model=LogisticRegression(random_state=0)    # buradan sonra modelleme


model.fit(X_train,y_train)

#şimdi tahmin işlemi

y_pred=model.predict(X_test)


#Performans Değerlendirmesi

#confisuon matris ile yapılır çünkü kesikli veri


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

confusion_matrix(y_true=y_test,y_pred=y_pred)#bunun sonucu bize bir 2 darray verir
classification_report(y_true=y_test,y_pred=y_pred)

accuracy_score(y_true=y_test,y_pred=y_pred)
print(classification_report(y_true=y_test,y_pred=y_pred))
print(accuracy_score(y_true=y_test,y_pred=y_pred))