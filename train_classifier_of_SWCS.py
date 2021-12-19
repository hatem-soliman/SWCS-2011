from helpers import defaultdict,Counter,bisect_right,log2,chi2,pickle
"""

node structure:

INTERNAL:
	[ (attribute, [branchVal_(1),branchVal_(2),...,branchVal_(N)] ), child_(1), child_(2), ..., child_(N+1)]
	If an element's attribute has a value >= to branchVal_(i) and < branchVal_(j), then it should be passed down to child_(i)

LEAF:s
	[(label1,attrvals), (label2,attrvals), ...]

"""

class DecisionTree():
	"""
	given a pointer to a node, recursively splits it until its entropy is below the maximum allowed entropy
	"""
	def grow(self,parent,pIndex):
		node=parent[pIndex]
		if isLeaf(node): # if it's a leaf node, try to split it
			entropy=getEntropy(node)
			if(entropy>self.maxEntropy):
				self.split(parent,pIndex,entropy)
			else:
				print 'leaf entropy is low enough:',str(entropy),'<=',str(self.maxEntropy)
		else: # if it's an internal node, recurse on all children
			for index in xrange(len(node)-1):
				self.grow(node,index+1)

	"""
	given a pointer to a node, and its current entropy, attempts to split it on the best possible attribute and value.
	the current implementation does a binary split, but this can easily be changed to support n-ary splits
	"""
	def split(self,parent,pIndex,minEntropy):
		node=parent[pIndex]
		print '\nAttempting to split leaf (size:',len(node),'entropy:'+str(minEntropy)+')'
		minTup=None
		for attr in xrange(len(node[0][1])): # for each attribute
			node.sort(key=lambda sample:sample[1][attr]) # sort the elements on that attribute
			currVal=node[0][1][attr]
			for index,(_,sample) in enumerate(node):
				newVal=sample[attr]
				if not newVal is currVal: # and every time the attribute's value changes we simulate a split and determine the subsequent total entropy
					currVal=newVal
					currEntropy,lChild,rChild=entropyBelow(minEntropy,node,attr,sample[attr],index)
					if currEntropy<minEntropy: # if the new entropy is lower than our previous min, save the split
						print 'split on attribute',self.attrNames[attr] if self.attrNames else attr,'at value',str(sample[attr])+': entropy='+str(currEntropy)
						minEntropy=currEntropy
						minTup=(attr,sample[attr],lChild,rChild)
		if not minTup:
			print 'Leaf is homogenous with value',map(lambda (attr,val): self.attrNames[attr]+': '+str(val),enumerate(node[0][1])) if self.attrNames else node[0][1],'and entropy',minEntropy
			return
		minAttr,splitpoint,leftChild,rightChild=minTup
		print 'size of children:'
		print '\tleft:',len(leftChild)
		print '\tright:',len(rightChild)
		pval=pvalue(node,[leftChild,rightChild])
		print 'pValue:',pval
		if pval>=self.maxP: # if our split isn't likely to lead to another statistically significant difference, we give up
			print 'Stop splitting!'
		else: # otherwise, perform the actual split and recurse
			parent[pIndex]=[(minAttr,[splitpoint]),leftChild,rightChild]
			self.grow(parent,pIndex)

	def insert(self,sample,node):
		if isLeaf(node):
			node.append(sample)
		else:
			splittribute,splitlist=node[0]
			insert(sample,node[1+bisect_right(splitlist,sample[1][splittribute])])

	def testFile(self,sampleFile,labelFile):
		samples=map(str.split,sampleFile.readlines())
		labels=map(str.strip,labelFile.readlines())
		sampleFile.close()
		labelFile.close()
		correct=0
		for index,sample in enumerate(samples):
			clas=classifySample(sample,self.tree[0])
			#print
			#print labels[index]
			#print clas
			if labels[index] == clas:
				#print 'correct!'
				correct+=1
		return correct/float(len(samples))

	def classifyFile(self,dataFile,dumpFile):
		for sample in map(str.split,dataFile.readlines()):
			dumpFile.write(classifySample(sample,self.tree[0]))

	"""
	trains a new tree by expanding the existing one or growing a new one
	"""
	def train(self,sampleFile,labelFile,attrNameFile,maxEntropy,maxP,maxSamples,maxAttrs):
		print 'train',(self,sampleFile,labelFile,attrNameFile,maxEntropy,maxP,maxSamples,maxAttrs)
		self.maxEntropy=maxEntropy
		self.maxP=maxP
		if attrNameFile:
			self.attrNames=map(str.strip,attrNameFile.readlines())
			#samples=zip(map(str.strip,labelFile.readlines()[:maxSamples]),[map(int,attrlist[:maxAttrs]) for attrlist in map(str.split,sampleFile.readlines()[:maxSamples])])
			samples=zip(map(str.strip,labelFile.readlines()[:maxSamples]),[attrlist[:maxAttrs] for attrlist in map(str.split,sampleFile.readlines()[:maxSamples])])
		else:
			self.attrNames=None
			samples=zip(map(str.strip,labelFile.readlines()[:maxSamples]),[line[:maxAttrs] for line in map(str.split,sampleFile.readlines()[:maxSamples])])
		sampleFile.close()
		labelFile.close()
		print 'numAttrs:',len(samples[0][1])
		try:
			for sample in samples:
				self.insert(sample,self.tree)
		except AttributeError:
			self.tree=[samples]
		self.grow(self.tree,0) # start the recursive splitting
		setClassifiers(self.tree,0) # finalize the tree
		print ('\n')
		print self.tostring(self.tree[0],0)
		print '\n'
		sizes=size(self.tree[0])
		print 'internal nodes:',sizes[0]
		print 'leaf nodes:',sizes[1]
		print 'total:',sum(sizes)
	def __str__(self):
		return self.tostring(self.tree[0],0)
	def tostring(self,tree,spaceCount):
		element=tree[0]
		if type(element) is str:
			return '-'*spaceCount+str(element)+'\n'
		attrIndex=element[0]
		return '-'*spaceCount+((self.attrNames[attrIndex]+'('+str(attrIndex)+')'+str(element[1])) if self.attrNames else str(element))+'\n'+self.tostring(tree[1],spaceCount+1)+self.tostring(tree[2],spaceCount+1)

"""
returns the number of internal and leaf nodes in a tree.
the tuple (3,5) means represents 3 internal nodes and 5 leaf nodes.
"""
def size(tree):
	if type(tree) is str:
		return (0,1)
	ls=tuple([size(child) for child in tree[1:]])
	ls=zip(*ls)
	return (1+sum(ls[0]),sum(ls[1]))

"""
sets a node equal to the most common classification label among its elements
"""
def setClassifiers(parent,index):
	node=parent[index]
	if isLeaf(node):
		parent[index]=Counter(element[0] for element in node).most_common(1)[0][0]
	else:
		for index in xrange(len(node)-1):
			setClassifiers(node,index+1)

def isLeaf(node):
	return type(node[0][0]) is str

def classifySample(sample,node):
	if type(node) is str:
		#print 'node:',node
		return node
	else:
		#return classifySample(sample,node[node[0][0](sample)])
		splittribute,splitlist=node[0]
		#print 'type(splittribute):',type(splittribute);print 'type(splitlist[0]):',type(splitlist[0]);print 'type(sample[splittribute]):',type(sample[splittribute]);print 'node[0]:',node[0];print 'sampleVal:',sample[splittribute];print 'splitlist:',splitlist;print
		return classifySample(sample,node[1+bisect_right(splitlist,sample[splittribute])])
		#return classifySample(sample,node[1 if sample[splittribute]<splitlist[0] else 2])

"""
determines the entropy of the specified *SORTED* node after undergoing a split on attrIndex at the value attrVal.
"""
def entropyBelow(maxEntropy,elements,attrIndex,attrVal,splitIndex):
	total=float(len(elements))
	#elements is already sorted so we just search for the first occurrence
	splitIndex=binaryFirstOccurence(attrIndex,attrVal,elements,0,splitIndex)
	#get left-child entropy
	subseqL=elements[:splitIndex]
	subtotal=len(subseqL)
	size=float(subtotal)
	counter=Counter([element[0] for element in subseqL])
	lEntropy=0
	for (_,val) in counter.iteritems():
		prob=val/size
		lEntropy-=prob*log2(prob)
	lEntropy*=subtotal/total
	if lEntropy>=maxEntropy: # if we've already surpassed maxEntropy, we fail early
		return (lEntropy,None,None)
	#get right-child entropy
	subseqR=elements[splitIndex:]
	subtotal=len(subseqR)
	size=float(subtotal)
	counter=Counter([element[0] for element in subseqR])
	rEntropy=0
	for (_,val) in counter.iteritems():
		prob=val/size
		rEntropy-=prob*log2(prob)
	return ((rEntropy*subtotal/total)+lEntropy,subseqL,subseqR)

"""
finds the first occurrence in elements where an element has value val at attribute attr
"""
def binaryFirstOccurence(attr,val,elements,low,high):
	while low<=high:
		mid=(low + high)>>1
		currVal=elements[mid][1][attr]
		if currVal < val:
			low = mid + 1
		elif currVal > val: 
			high = mid - 1
		else:
			if mid != 0 and elements[mid-1][1][attr] is val:
				high=mid-1
			else:
				return mid

"""
derives the p-value of splitting parent into leaves using the chi-squared test
"""
def pvalue(parent,leaves):
	#derive chi-squared statistic
	chiT=0
	n=float(len(parent))
	parentCounter=Counter(element[0] for element in parent)
	for leaf in leaves:
		ratio=len(leaf)/n
		for attr,leafCount in Counter(element[0] for element in leaf).iteritems():
			expected=parentCounter[attr]*ratio
			chiT+=((expected-leafCount)**2)/expected
	print 'chiT:',chiT
	#derive p-value
	return chi2.sf(chiT,len(parentCounter)-1)

"""
determines the entropy of elements with respect to their classification labels
"""
def getEntropy(elements):
	size=float(len(elements))
	counter=Counter(element[0] for element in elements)
	entropy=0
	for _,val in counter.iteritems():
		prob=val/size
		entropy-=prob*log2(prob)
	return entropy

def load(treeFile):
	tree=DecisionTree()
	tree.tree=pickle.load(treeFile)
	return tree

def save(tree,treeFile):
	pickle.dump(tree.tree,treeFile)







------------------------------


import decisiontree
from helpers import defaultdict,argv,stdout,join,getcwd

def printErr():
	execName=argv[0]
	print 'Usage: python',execName,'--option value --otheroption othervalue [...]'
	print
	print 'default options: --train WORKING_DIR --test WORKING_DIR'
	print
	print 'Options:'
	print '--load tree_file'
	print '\tload the specified tree'
	print '--train directory (chainable)'
	print '\ttrain the tree; directory must contain trainData.csv and trainLabels.csv; if attrNames.csv exists, it will be used to convert the attribute numbers into their respective names when printed'
	print '\tif a tree is already loaded or trained, training data will be inserted and then the tree will be trained from that point'
	print '\totherwise, a new tree will be grown from the training data'
	print '--maxEntropy val'
	print '\tset the maximum entropy allowed in the tree after each training (default: 0)'
	print '--maxPValue val'
	print '\tset the maximum p-value for which a node will be split (default: 1)'
	print '--maxSamples val'
	print '\tcap the input at val samples maximum'
	print '--maxAttrs val'
	print '\tcap the input at val attributes per sample'
	print '--save_tree file'
	print '\tsave the tree in the specified file; save_tree def saves the tree in the working directory'
	print '--save_tree_ascii file'
	print '\tsave the ascii display of the tree in the specified file; save_tree_ascii def saves the tree in the working directory'
	print '--test directory (chainable)'
	print '\ttest the tree on testing data; directory must contain testData.csv and testLabels.csv'
	print '--classify file (chainable)'
	print '\tuse the tree to classify the data in file'
	print '--save_classifications file'
	print '\tappend the classification results to the specified file; save_classifications def saves the output in the working directory (default: stdout)'

args=defaultdict(list)
try:
	for index in xrange(1,len(argv),2):
		args[argv[index][2:]].append(argv[index+1])
except:
	printErr()
	quit()

arglist=['load','train','maxEntropy','maxPValue','maxSamples','maxAttrs','save_tree','save_tree_ascii','test','classify','save_classifications']
for arg in args:
	if not arg in arglist:
		print 'Invalid option:',arg
		printErr()
		quit()
if 'maxEntropy' in args:
	maxEnt=float(args['maxEntropy'][0])
else:
	maxEnt=0.0

if 'maxPValue' in args:
	maxPValue=float(args['maxPValue'][0])
else:
	maxPValue=1.0

if 'maxSamples' in args:
	maxSamples=int(args['maxSamples'][0])
else:
	maxSamples=None

if 'maxAttrs' in args:
	maxAttrs=int(args['maxAttrs'][0])
else:
	maxAttrs=None

def trainTree(tDir='.'):
	tFile=open(join(tDir,'trainData.csv'),'r')
	lFile=open(join(tDir,'trainLabels.csv'),'r')
	try:
		nFile=open(join(tDir,'attrNames.csv'),'r')
	except:
		nFile=None
	print tDir+':'
	tree.train(tFile,lFile,nFile,maxEnt,maxPValue,maxSamples,maxAttrs)

def saveTree(treefile=join(getcwd(),'tree_maxEnt'+str(maxEnt)+'_maxP'+str(maxPValue)+'_maxSam'+str(maxSamples)+'_maxAttrs'+str(maxAttrs))):
	with open(treefile,'wb') as saveF:
		decisiontree.save(tree,saveF)

def testTree(tDir='.'):
	with open(join(tDir,'testData.csv'),'r') as dataFile:
		with open(join(tDir,'testLabels.csv'),'r') as labelFile:
			print tDir+':',tree.testFile(dataFile,labelFile)

if 'load' in args:
	filename=args['load'][0]
	try:
		with open(filename,'rb') as treeFile:
			tree=decisiontree.load(treeFile)
	except:
		print filename,'could not be loaded!'
		quit()
else:
	print '\nTRAINING\n'
	tree=decisiontree.DecisionTree()
	if 'train' in args:
		for tDir in args['train']:
			trainTree(tDir)
	else:
		tree=decisiontree.DecisionTree()
		trainTree()
		saveTree()
		if not 'test' in args:
			print '\nTESTING\n'
			testTree()

if 'save_tree' in args:
	filename=args['save_tree'][0]
	if filename == 'def':
		saveTree()
	else:
		saveTree(filename)

if 'save_tree_ascii' in args:
	filename=args['save_tree_ascii'][0]
	with (open(join(getcwd(),'tree_maxEnt'+str(maxEnt)+'_maxP'+str(maxPValue)+'_maxSam'+str(maxSamples)+'_maxAttrs'+str(maxAttrs)+'_ASCII.txt'),'w') if filename == 'def' else open(filename,'w')) as saveFile:
		saveFile.write(str(tree))

if 'test' in args:
	print '\nTESTING\n'
	for testDir in args['test']:
		testTree(testDir)

if 'classify' in args:
	print '\nCLASSIFYING\n'
	filename=args['save_classifications'][0]
	if filename == 'def':
		filename=join(getcwd(),'classification.txt')
	with (open(filename,'a') if 'save_classifications' in args else stdout) as saveFile:
		for classFile in args['classify']:
			with open(classFile,'r') as classFile:
				tree.classifyFile(classFile,saveFile)
