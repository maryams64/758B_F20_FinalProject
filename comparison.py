import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

#Data Cleaning
df = pd.read_json('data/Digital_Music_5.json', lines = True)

df['reviewText_item'] = df.groupby(['asin'])['reviewText'].transform(lambda x : ' '.join(x)) 
df['overall_item']=df.groupby('asin').overall.transform('mean').round()
df['reviewText_user'] = df.groupby(['reviewerID'])['reviewText'].transform(lambda x : ' '.join(x)) 
df['overall_user']=df.groupby('reviewerID').overall.transform('mean').round()

df['reviewText_item'].dropna(inplace=True)
df['reviewText_user'].dropna(inplace=True)

df=df.drop_duplicates(subset=['reviewText_item'])
df=df.drop_duplicates(subset=['reviewText_user'])

cols = ['overall_item','overall_user']
df['overall_avg'] = df[cols].astype(float).mean(axis=1)
df['overall_avg'] = df['overall_avg'].round()
df['overall_avg'] = df['overall_avg'].astype(int)

df1=df.sample(frac=0.5)

zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
df1['overall'] = df1['overall'].apply(lambda x: zero_numbering[x])
df1['overall_avg'] = df1['overall_avg'].apply(lambda x: zero_numbering[x])
df1['all_text']=df1['reviewText_item']+df1['reviewText_user']

#Tf-idf Vectorization
vectorizer = TfidfVectorizer(max_features=200,stop_words='english')
X_all=vectorizer.fit_transform(df1['all_text']) 
Y_overall=np.array(df1['overall_avg'])

#Random Forest Model
kfold = KFold(n_splits=10,shuffle=True,random_state=2020)

rf_cvscores = [] # store accuracy score for each fold
rf_model = RandomForestClassifier(random_state=2020,max_depth=2,criterion='entropy')

for train, test in kfold.split(X_all):
  rf_model.fit(X_all[train],Y_overall[train])
  rf_acc = rf_model.score(X_all[test],Y_overall[test])
  
  rf_cvscores.append(rf_acc)

print("Random Forest Accuracy Score: %.4f%% (std: +/- %.6f%%)" % (np.mean(rf_cvscores)*100, np.std(rf_cvscores)*100))

#SVM
X_train, X_test, Y_train, Y_test=train_test_split(X_all,Y_overall,test_size=0.3)
clf=SVC(gamma="auto")
clf.fit(X_train,Y_train)
predict=clf.predict(X_test)
print(f"SVM Accuracy Score: {metrics.accuracy_score(Y_test,predict)*100:0.4f}%")
