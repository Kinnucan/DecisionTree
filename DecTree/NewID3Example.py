from DecTree.corpus import Corpus
from DecTree.DTree import DecisionTree

corp = Corpus("PythonDatasets/dog2.dat")
dTree = DecisionTree(corp)

dTree.buildTree()
dTree.printTree()