# Import Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fertility.csv')
# print(df.head()) # check dataframe
# print(df.columns.values) # chech columns value
# print(df.info()) # check dataframe info

'''
check value counts : jumlah data unique
'''
# print(df['Season'].value_counts())
# print('\n')
# print(df['Childish diseases'].value_counts())
# print('\n')
# print(df['Accident or serious trauma'].value_counts())
# print('\n')
# print(df['Surgical intervention'].value_counts())
# print('\n')
# print(df['High fevers in the last year'].value_counts())
# print('\n')
# print(df['Frequency of alcohol consumption'].value_counts())
# print('\n')
# print(df['Smoking habit'].value_counts())
# print('\n')
# print(df['Diagnosis'].value_counts())

'''
get_dummies change value
'''
child_diseases = pd.get_dummies(df['Childish diseases'],drop_first=True)
accident = pd.get_dummies(df['Accident or serious trauma'],drop_first=True)
surgical_intv = pd.get_dummies(df['Surgical intervention'],drop_first=True)
high_fever = pd.get_dummies(df['High fevers in the last year'],drop_first=True)
alcohol_cons = pd.get_dummies(df['Frequency of alcohol consumption'],drop_first=True)
smoking = pd.get_dummies(df['Smoking habit'],drop_first=True)
diagnosis = pd.get_dummies(df['Diagnosis'],drop_first=True)

# make new dataframe and change columns name
df.drop(['Season', 'Childish diseases', 'Accident or serious trauma',
       'Surgical intervention', 'High fevers in the last year',
       'Frequency of alcohol consumption', 'Smoking habit','Diagnosis'],axis=1,inplace=True)

df = pd.concat([df, child_diseases, accident, surgical_intv, high_fever, alcohol_cons, smoking, diagnosis],axis=1)

df.columns = [
    'Age', 'Number of hours spent sitting per day', 
    'childish diseases', 'accident', 'surgical', 'more than 3 months ago', 'no',
    'hardly ever or never', 'once a week', 'several times a day',
    'several times a week', 'never', 'occasional', 'Target']

# print(df.head())
# print(df.columns.values)


'''
Exploratory Data Analysis
'''

# checking target data
# sns.countplot(df['Target']) # data nya tidak balance
# plt.show()

# train test split
from sklearn.model_selection import train_test_split

X = df.drop('Target',axis=1)
y = df['Target']

# model Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logistic_model = LogisticRegression(solver='liblinear')
svm_model = SVC(gamma='auto')
knn_model = KNeighborsClassifier(n_neighbors=2)

# resampling dataset: untuk mengubah menjadi balance
count_class_0, count_class_1 = df['Target'].value_counts()
df_class_0 = df[df['Target']==0] 
df_class_1 = df[df['Target']==1]

df_class_1_under = df_class_1.sample(count_class_1) 
df_test_under = pd.concat([df_class_0,df_class_1_under],axis=0)

# print('Random under-sampling')
# print(df_test_under['Target'].value_counts())

# data visualization cek balance data target
# sns.countplot(df_test_under['Target'])
# plt.show()


'''
Create Machine Learning Model
'''
X = df_test_under.drop('Target',axis=1)
y = df_test_under['Target']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,random_state=101)

# fitting model
logistic_model.fit(X_train,y_train)
svm_model.fit(X_train,y_train)
knn_model.fit(X_train,y_train)

# predict using balance target
logistic_prediction = logistic_model.predict(X_test)
svm_prediction = svm_model.predict(X_test)
knn_prediction = knn_model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

'''
prediction score
'''
# print('Logistic Regression Report')
# print(classification_report(y_test,logistic_prediction))
# print(confusion_matrix(y_test,logistic_prediction))
# print(knn_model.score(X_test,y_test))
# print('\n')

# print('SVM Report')
# print(classification_report(y_test,svm_prediction))
# print(confusion_matrix(y_test,svm_prediction))
# print(svm_model.score(X_test,y_test))
# print('\n')

# print('KNN Report')
# print(classification_report(y_test,knn_prediction))
# print(confusion_matrix(y_test,knn_prediction))
# print(knn_model.score(X_test,y_test))
# print('\n')

'''
Column Predictions =
'Age' => age (1)
'Number of hours spent sitting per day' => sit hour (1)
'childish diseases' => child diseases (1)
'accident' => accident (1)
'surgical' => surgery (1)
'more than 3 months ago', 'no' => fever (2)
'hardly ever or never','once a week', 'several times a day', 'several times a week' => alcohol (4)
'never', 'occasional' => smoke (2)
'''
# Prediksi Arin (Montir, 29 th) :
# Sejak kecil terkenal sehat & lincah, tak pernah mengalami penyakit serius. 
# Usai menjadi Sarjana Teknik, Arin meneruskan usaha bengkel ayahnya. 
# Setiap hari menghabiskan 5 jam untuk duduk, sembari merokok & mengkonsumsi alkohol.

arin = [np.array([29, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])]

# Bebi (Chef, 31 th):
# Memutuskan fokus menggeluti bidang kuliner setelah 10 tahun yang lalu kakinya terpaksa diamputasi lantaran kecelakaan lalu lintas. 
# Tidak merokok 
# namun dalam seminggu beberapa kali mengkonsumsi alkohol.

bebi = [np.array([31, 5, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0])]

# Caca (Gardener, 25 th)
# Pecinta lingkungan yang terobsesi dengan gaya hidup sehat. 
# Sayangnya daya tahan tubuhnya lemah. 
# Sedari kecil hingga kini, Caca kerap kali terjangkit penyakit, terutama batuk, pilek & demam. 
# Dalam sehari, 7 jam ia habiskan untuk duduk.

caca = [np.array([25, 7, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])]

# Dini (Dosen, 28 th)
# Dosen muda ini 2 bulan lalu baru saja menjalani operasi patah tulang rusuk, akibat cedera saat berolahraga. 
# Kini ia terpaksa duduk di kursi roda, selama masih dalam masa penyembuhan hingga 1 bulan ke depan. 
# Setiap hari Dini merokok, namun sangat anti pada alkohol.

dini = [np.array([28, 24, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])]

# Enno (Dokter, 42 th)
# Semasa kecil, Enno kerap kali terjangkit asma akut. Bahkan pernah menjalani perawatan intensif akibat bronkitis. 
# Kini sebagai dokter umum, ia senantiasa menjaga kebersihan & kesehatan. 
# Dalam sehari, 8 jam ia habiskan untuk melayani konsultasi pasien di poli umum.

enno = [np.array([42, 8, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0])]

def klasifikasi_kesuburan(prediksi):
    if prediksi[0]==0:
        return 'Altered'
    else:
        return 'Normal'

karakter_sample = [arin,bebi,caca,dini,enno]
nama_sample = ['Arin','Bebi','Caca','Dini','Enno']

for i in  range(len(karakter_sample)):
    print(f'{nama_sample[i]}, prediksi kesuburan:',klasifikasi_kesuburan(logistic_model.predict(karakter_sample[i])),'(Logistic Regression)')
    print(f'{nama_sample[i]}, prediksi kesuburan:',klasifikasi_kesuburan(svm_model.predict(karakter_sample[i])),'(Support Vector Machines)')
    print(f'{nama_sample[i]}, prediksi kesuburan:',klasifikasi_kesuburan(knn_model.predict(karakter_sample[i])),'(K-Nearest Neighbors)')
    print('\n')