
import numpy as np

from math import log

import operator

 

class TreeNode:

	# class label for leaf node

	label = None

	# split feat name for branching node

	feat = None

	# subTree dict {feat value : sub tree}

	subTrees = {}

	

	def __init__(self):

		self.label = None

		self.feat = None

		self.subTrees = {}

	

	""" add sub tree node to this tree node 

	

	Parameters

    ----------

	feat : feat value of this sub tree branch 

	subTree : sub tree node

	"""	

	def addSubTree(self, feat, subTree):

		print 'subTree to add is %s, subTrees is %s}' % (feat, self.subTrees)

		if feat in self.subTrees:

			raise Exception('duplicate feat')

		self.subTrees[feat] = subTree

 

class DecisionTree:

	def __init__(self):

		# a decision tree

		tree = None	

		# list of feat name

		feats = None

	

	""" calculate Shannon Entropy 

	

	Parameters

    ----------

	labels : labels for the data



	Returns

    -------

	sum : Shannon Entropy

	"""

	def shannonEnt(self, labels):

		labelDict = {}

		for label in labels:

			labelDict[label] = labelDict.get(label, 0) + 1

		

		sum = 0

		length = len(labels)

		for value in labelDict.itervalues():

			prob = float(value) / length

			sum -= prob * log(prob, 2)

			

		return sum

	

	""" calculate majority label for the data 

	

	Parameters

    ----------

	labels : labels for the data



	Returns

    -------

	majority label

	"""

	def majorityCount(self, labels):

		labelCount = {}

		for label in labels:

			labelCount[label] = labelCount.get(label, 0) + 1

		print 'majority count is %s' % (labelCount)

		sortedLabelCount = sorted(labelCount.iteritems(), key = operator.itemgetter(1), reverse = True)

		return sortedLabelCount[0][0]

		

	def splitDataSet(self, dataSet, labels, featIndex, value):

		dataSetSlice = []

		labelSlice = []

		num = len(labels)

		for i in range(num):

			if dataSet[i][featIndex] == value:

				leftData = dataSet[i][:featIndex]

				leftData.extend(dataSet[i][featIndex+1:])

				dataSetSlice.append(leftData)

				labelSlice.append(labels[i])

		return dataSetSlice, labelSlice

	

	""" choose best feat to split data 

	

	Parameters

    ----------

	dataSet : 

	labels : list of label for the data



	Returns

    -------

	bestFeat : best feat index to split the data

	"""

	def chooseBestFeatToSplit(self, dataSet, labels):

		bestFeat = -1

		minEnt = float("inf")

		numFeat = len(dataSet[0])

		for i in range(numFeat):

			uniqueFeats = set(data[i] for data in dataSet)

			entropy = 0

			for feat in uniqueFeats:

				labelSlice = [labels[i] for data in dataSet if data[i] == feat]

				prob = len(labelSlice) / float(len(labels))

				entropy += prob * self.shannonEnt(labelSlice) 

			if entropy < minEnt:

				minEnt = entropy

				bestFeat = i

		return bestFeat

		

	""" create a tree using data and their labels 

	

	Parameters

    ----------

	dataSet : 

	labels : list of label for the data

	feats : list of name for each feat



	Returns

    -------

	treeNode : root tree node of the tree

	"""

	def createTree(self, dataSet, labels, feats):

		print 'create tree'

		print dataSet

		print labels

		print feats

		treeNode = TreeNode()

		print 'new tree node, subTrees is %s' % (treeNode.subTrees)

		if len(dataSet[0]) == 0:

			treeNode.label = self.majorityCount(labels)

			print 'get leaf node, label is %s' % (treeNode.label)

			return treeNode

		if labels.count(labels[0]) == len(labels):

			print 'get leaf node, label is %s' % (labels[0])

			treeNode.label = labels[0]

			return treeNode

		bestFeat = self.chooseBestFeatToSplit(dataSet, labels)

		treeNode.feat = feats[bestFeat]

		featValues = set([data[bestFeat] for data in dataSet])

		

		for featValue in featValues:

			print 'featValue is %s : %s' % (feats[bestFeat], featValue)

			remainingFeats = feats[:]

			del(remainingFeats[bestFeat])

			dataSetSlice, labelSlice = self.splitDataSet(dataSet, labels, bestFeat, featValue)

			subTree = self.createTree(dataSetSlice, labelSlice, remainingFeats)

			treeNode.addSubTree(featValue, subTree)

		

		return treeNode

		

	def storeTree(self, fileName):

		import pickle

		fw = open(fileName, 'w')

		pickle.dump(self.tree, fw)

		fw.close()

		

	def loadTree(self, fileName):

		import pickle

		fw = open(fileName)

		self.tree = pickle.load(fw)

		fw.close()

 

	""" train a tree using data and their labels 

	

	Parameters

    ----------

	dataSet : 

	labels : list of label for the data

	feats : list of name for each feat

	"""	

	def train(self, dataSet, labels, feats):

		self.feats = feats

		self.tree = self.createTree(dataSet, labels, feats)

	

	""" classify a piece of data using decision tree 

	

	Parameters

    ----------

	data : a list

	

	Returns

    -------

	input data's label

	"""	

	def classify(self, data):

		treeNode = self.tree

		while (True) :

			if (treeNode.label != None):

				return treeNode.label

			featIndex = self.feats.index(treeNode.feat)

			treeNode = treeNode.subTrees[data[featIndex]]

			
