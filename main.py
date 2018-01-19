from __future__ import print_function

from Utils import Utils
from Tree import node
from Boosting import Boosting
from Logic import Prover
from sys import argv

class GradientBoosting(object):
    '''class to perform gradient boosting'''

    def __init__(self,advice=False,regression=False,trees = 10,treeDepth = 2):
        '''class constructor'''
        self.targets = None
        self.advice = advice
        self.regression = regression
        self.adviceClause = None
        self.numberOfTrees = trees
        self.treeDepth = treeDepth
        self.trees = {}
        self.data = None
        self.testPos = {}
        self.testNeg = {}
        self.probabilisticKnowledgeFacts = {}

    def setTargets(self,targets):
        '''sets targets to learn'''
        self.targets = targets

    def setAdviceClause(self,adviceClause):
        '''sets advice clauses'''
        self.adviceClause = adviceClause

    def getModifiedBkFile(self,knowledgeModels):
        '''returns modified bk file'''
        with open("train/bk.txt") as fp:
            bk = fp.read().splitlines()
            bkWithTarget = [line for line in bk if '+' not in line and '-' not in line]
            arguments = bkWithTarget[0][:-1].split('(')[1]
            for knowledgeModel in knowledgeModels:
                literalName = "knowledge"+str(knowledgeModel.ID)
                knowledgeBk = literalName+'('+arguments+')'
                knowledgeBk = knowledgeBk.replace("(","(+")
                knowledgeBk = knowledgeBk.replace(",",",+")
                bk.append(knowledgeBk)
            return bk

    def constructKnowledgeFacts(self,data,knowledgeModels):
        '''construct knowledge facts for 2nd layer'''
        bk = self.getModifiedBkFile(knowledgeModels)
        data.setBackground(bk)
        allExamples = list(data.pos.keys())+list(data.neg.keys())
        for knowledgeModel in knowledgeModels:
            self.probabilisticKnowledgeFacts[knowledgeModel] = {}
            for example in allExamples:
                if knowledgeModel.applies(data,example):
                    target = example.split('(')[0]
                    trees = knowledgeModel.clf.trees[target]
                    sumOfGradients = Boosting.computeSumOfGradients(example,trees,data)
                    probabilityOfExample = Utils.sigmoid(Boosting.logPrior+sumOfGradients)
                    self.probabilisticKnowledgeFacts[knowledgeModel][example] = probabilityOfExample
                    if probabilityOfExample > 0.5:
                        args = example[:-1].split('(')[1]
                        literalName = "knowledge"+str(knowledgeModel.ID)
                        data.addFact(literalName+'('+args+')')

    def learn(self,knowledge=False,makeKnowledgeFacts=False,knowledgeModels=False):
        '''method to run gradient boosting'''
        for target in self.targets:
            data = Utils.readTrainingData(target)
            if knowledge:
                adviceClause = knowledge.clause
                data.knowledge = True
                data.adviceClauses[adviceClause] = {}
                data.adviceClauses[adviceClause]['preferred'] = knowledge.preferredTargets
                data.adviceClauses[adviceClause]['nonPreferred'] = knowledge.nonPreferredTargets
            if knowledgeModels:
                self.constructKnowledgeFacts(data,knowledgeModels)
            trees = []
            for i in range(self.numberOfTrees):
                print('='*20,"learning tree",str(i),'='*20)
                node.setMaxDepth(self.treeDepth)
                node.learnTree(data)
                trees.append(node.learnedDecisionTree)
                Boosting.updateGradients(data,trees)
            self.trees[target] = trees
            for tree in trees:
                print('='*30,"tree",str(trees.index(tree)),'='*30)
                for clause in tree:
                    print(clause)
                    
    def infer(self,knowledgeModels = False):
        '''performs testing'''
        for target in self.targets:
            testData = Utils.readTestData(target,self.regression)
            if knowledgeModels:
                self.constructKnowledgeFacts(testData,knowledgeModels)
            print (self.trees[target])
            Boosting.performInference(testData,self.trees[target])
            self.testPos[target] = testData.pos
            self.testNeg[target] = testData.neg
            print (testData.pos)
            print (testData.neg)

class Knowledge(object):
    '''stores model for each piece of knowledge'''

    def __init__(self,ID,clause,preferredTargets,nonPreferredTargets):
        '''constructor'''
        self.ID = ID
        self.clause = clause
        self.preferredTargets = preferredTargets
        self.nonPreferredTargets = nonPreferredTargets
        self.clf = None
        self.learn()

    def applies(self,data,example):
        '''checks if knowledge applies to example'''
        if Prover.prove(data,example,self.clause):
            return True

    def learn(self):
        '''learns a model based on the advice clause'''
        self.clf = GradientBoosting()
        self.clf.setTargets(["putdown"])
        self.clf.learn(knowledge = self)

    def infer(self):
        '''performs testing with knowledge'''
        self.clf.infer()            

def main():
    '''main method'''
    knowledge1 = Knowledge(1,"putdown(F):-on(F,H,Z),ontable(F,Z,table)",["putdown"],[])
    knowledge2 = Knowledge(2,"putdown(F):-on(F,H,Z),ontable(F,Z,table)",["putdown"],[])
    #second layer
    clf = GradientBoosting()
    clf.setTargets(["putdown"])
    clf.learn(makeKnowledgeFacts = True,knowledgeModels = [knowledge1,knowledge2])
    clf.infer(knowledgeModels = [knowledge1,knowledge2])

main()
    
'''
def main():
    #main method
    targets = argv[argv.index("-target")+1][1:-1].split(',') #read targets from input
    regression,advice = False,False
    if "-reg" in argv:
        regression = True
    if "-expAdvice" in argv:
        advice = True
    for target in targets:
        data = Utils.readTrainingData(target,regression,advice) #read training data
        numberOfTrees = 10 #number of trees for boosting
        trees = [] #initialize place holder for trees
        for i in range(numberOfTrees): #learn each tree and update gradient
            print('='*20,"learning tree",str(i),'='*20)
            node.setMaxDepth(2)
            node.learnTree(data) #learn RRT
            trees.append(node.learnedDecisionTree)
            Boosting.updateGradients(data,trees)
        for tree in trees:
            print('='*30,"tree",str(trees.index(tree)),'='*30)
            for clause in tree:
                print(clause)
        testData = Utils.readTestData(target,regression) #read testing data
        Boosting.performInference(testData,trees) #get probability of test examples
        
        #print testData.pos #--> uncomment to see test query probabilities (for classification)
        #print testData.neg

        #print testData.examples #--> uncomment to see test example values (for regression)

        
main()
'''
