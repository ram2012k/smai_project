import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from random import seed
from random import randrange , randint , shuffle
from numpy import sqrt
def split_data(dataset):
	testSet=[]
	ratio=0.1
	testSize=int(ratio*len(dataset))
	i=0
	while i < testSize:
		index=randint(0,len(dataset)-1)
		testSet.append(dataset[index])
		del dataset[index]
		i=i+1
	return [dataset,testSet]

def load_training_data(training_file_name):
	dataset=[]
	with open(training_file_name) as f:
		for line in f:
			line=line.split(" ")
			feature_Vector=line[1:-1]
			features=[]
			for feature in feature_Vector:
				features.append(feature.split(":")[1])
			features.append(line[0])
			dataset.append(features)
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] <= value:
			left.append(row)
		else:
			right.append(row)
	return left, right
 
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini
 
def get_split(dataset,indexes):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1) :
		if index in indexes:
			continue
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	indexes.append(b_index)
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
def split(node, max_depth, min_size, depth,indexes):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		list2=list(indexes)
		node['left'] = get_split(left,list2)
		split(node['left'], max_depth, min_size, depth+1,list2)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		list2=list(indexes)
		node['right'] = get_split(right,list2)
		split(node['right'], max_depth, min_size, depth+1,list2)

def build_tree(train, max_depth, min_size,indexes):
	root = get_split(train,indexes)
	#print "root " , root
	split(root, max_depth, min_size, 1,indexes)
	return root

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def plot_data(depth,accuracies):
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)

	ax.scatter(depth,accuracies)
	ax.semilogx(depth,accuracies,basex=2)
	ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
	ax.set_xlabel("depth")
	ax.set_title("graph")
	ax.set_ylabel("accuracy")
	ax.set_ylim([0,100])
	plt.show()

dataset=load_training_data("modifiedtrainingdata.svm")
shuffle(dataset)
dataset=dataset[:400]
li=[]
for i in dataset:
	li.append(i[-1])
print li
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset,i)
for i in range(len(dataset)):
	dataset[i][-1]=int(dataset[i][-1])

	if dataset[i][-1]==0 or dataset[i][-1]==1:
		dataset[i][-1]=0
	else:
		dataset[i][-1]=1


dataset,testSet=split_data(dataset)
depth=2
print "depth ", depth
depths=[]
accuracies=[]
while depth <= 100:
	print len(dataset[0])
	print depth
	indexes=[]
	tree=build_tree(dataset,depth,10,indexes)
	print len(indexes)
	print tree
	

	predictions=[]
	for row in testSet:
		prediction=predict(tree,row)
		predictions.append(prediction)
	count=0
	testlist=[]
	for i in testSet:
		testlist.append(i[-1])
	print testlist
	print predictions
	for i in range(0,len(predictions)):
		if int(predictions[i])==int(testSet[i][-1]):
			count=count+1;
	print len(predictions)

	print float(count)*100/len(predictions)
	depths.append(depth)
	accuracies.append(float(count)*100/len(predictions))
	depth=depth*2

plot_data(depths,accuracies);
