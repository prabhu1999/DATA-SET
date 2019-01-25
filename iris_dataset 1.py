#!/usr/bin/env python
# coding: utf-8

# In[1]:


#python version
import sys
print('python:{}'.format(sys.version))


# In[3]:


#scipy
import scipy
print('scipy:{}'.format(scipy.__version__))


# In[4]:


#numpy
import numpy
print('numpy:{}'.format(numpy.__version__))


# In[5]:


#matplotlib
import matplotlib
print('matplotlib:{}'.format(matplotlib.__version__))


# In[6]:


#pandas
import pandas
print('pandas:{}'.format(pandas.__version__))


# In[7]:


#scikit learn
import sklearn
print('sklearn:{}'.format(sklearn.__version__))


# In[42]:


import seaborn
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[58]:


names = ['sepal-length', 'sepal-width', 'petal-width', 'class']
dataset = pandas.read_csv('iris.csv', names=names)


# In[30]:


print(dataset.head(5))


# In[31]:


print(dataset.shape)


# In[32]:


print(dataset.describe())


# In[33]:


dataset = dataset.drop('Id',axis=1)


# In[34]:


print(dataset.head(5))


# In[63]:


print(dataset.head(20))


# In[35]:


print(dataset.describe())


# In[61]:


print(dataset.groupby('class').size())


# In[11]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[12]:


seaborn.pairplot(dataset, hue="Species", size=3, diag_kind="kde")
plt.show()


# In[14]:


seaborn.pairplot(dataset, hue="Species", size = 3)
seaborn.set()


# In[15]:


dataset.hist()
plt.show()


# In[44]:


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[45]:


num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'


# In[43]:


scatter_matrix(dataset)
plt.show()


# In[53]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[62]:


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklables(names)
plt.show()


# In[56]:


#predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[4]:


import numpy
X_new = numpy.array([[3, 2, 4, 0.2], [ 4.7, 3, 1.3, 0.2 ]])
print("X_new.shape:{}".format(X_new.shape))


# In[ ]:




