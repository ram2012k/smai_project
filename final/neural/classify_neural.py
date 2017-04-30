import csv
import numpy as np
import matplotlib.pyplot as plt,scipy, pylab
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from random import randint
from sklearn.preprocessing import Imputer
import random
def split(x,y,num):
	z_x=[]
	z_y=[]
	X=[]
	Y=[]
	#x=x.tolist()
	#y=y.tolist()
	ratio=0.1
	start=len(y)*ratio*num
	end=len(y)*ratio*(num+1)
	#testsize=int(ratio*len(y))
	for i in range(0,len(y)):
		if i>=start and i<end:
			z_x.append(x[i])
			z_y.append(y[i])
		else:
			X.append(x[i])
			Y.append(y[i])
	X=np.asarray(X)
	Y=np.asarray(Y)
	z_x=np.asarray(z_x)
	z_y=np.asarray(z_y)
	return [z_x,z_y,X,Y]
fname="modifiedtrainingdata.svm"
with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content]
random.shuffle(content)
y=[]
x=[]
files = open("testfile.txt","w")
for i in range(0,len(content)):
    temp=content[i].split(' ')
    if int(temp[0]) > 1:
        y.append(0)
    else:
        y.append(1)
    tup=[]
    for j in range(1,len(temp)):
        t1=temp[j].split(":")
        test=float(t1[1])
        tup.append(test)
    x.append(tup)

#y=np.asarray(y)
#X=np.asarray(x)
#split experiment data into 10% test and 90% train

#print len(x)
#print len(y)
#z_x,z_y=split(x,y)
#print len(x)
#print len(y)
#y=np.asarray(y)
#X=np.asarray(x)
# clf = MLPClassifier(activation='logistic', batch_size='auto',
#        beta_1=0.9, beta_2=0.999, early_stopping=False,
#        epsilon=1e-08, hidden_layer_sizes=(50,50), learning_rate='adaptive',
#        learning_rate_init=0.001, max_iter=500, momentum=0.9,
#        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#        warm_start=False)
#print np.isnan(X.any())
imp=Imputer(missing_values='NaN', strategy='median', axis=0)
#imp.fit(X)
x = imp.fit_transform(x)
#print np.isfinite(X.all())
#print len(X)
#print len(y)
#print len(z_x)
#print len(z_y)
#for i in range(0,len(X)):
#	print np.isnan(X[i].any())
#clf = svm.SVC(C=5000, cache_size=200, gamma=3, kernel='rbf',
#    max_iter=-1,tol=0.00001, verbose=False)
avg=0.0
count=0
for i in range(0,10):
	z_x,z_y,X,Y=split(x,y,i)
	clf = MLPClassifier(solver='sgd',activation='logistic', hidden_layer_sizes=(25,25),tol=0.0001)
	clf.fit(X,Y)
	print clf.score(z_x,z_y)
	avg+=clf.score(z_x,z_y)
avg/=10
print "accuracy",
print avg