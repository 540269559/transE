# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import copy
import time


# 使用的数据集
datasetName = ""

# 将进程运行的内容文件删除
def clearProgressText(filePath='./progressMessage/', fileName='trainStatus.txt'):
    file = filePath + fileName
    if os.path.exists(file):
        os.remove(file)

# 将进程运行的内容输入到text中
def writeProgressText(text, filePath='./progressMessage/', fileName='trainStatus.txt'):
    print(text)
    file = filePath + fileName
    with open(file, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

# 存储结果
def storeResult(fileName, vectorDict, filePath='./trainResult/'):
    file = filePath + fileName
    with open(file, 'w', encoding='utf-8') as f:
        for key in vectorDict:
            f.write(key + "\t")
            f.write(str(list(vectorDict[key])))
            f.write("\n")

# 数据加载
def dataForIdLoader(datasetPath, fileName):
    fullFileName = datasetPath + fileName
    objName2objId = {}
    with open(fullFileName, 'r') as file:
        lines = file.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            if len(arr) != 2:
                continue
            objectName = arr[0]
            objectId = arr[1]
            objName2objId[objectName] = objectId
    return objName2objId

def dataForTripleLoader(datasetPath, fileName, entity2id, relation2id):
    fullFileName = datasetPath + fileName
    tripleList = []
    with open(fullFileName, 'r') as file:
        lines = file.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            headEntityName = arr[0]
            tailEntityName = arr[2]
            relationName = arr[1]
            
            headEntityId = entity2id[headEntityName]
            tailEntityId = entity2id[tailEntityName]
            relationId = relation2id[relationName]
            tripleList.append([headEntityId, relationId, tailEntityId])
    return tripleList

def dataLoader(datasetPath):
    # 0.记录当前数据集的名称
    datasetName = datasetPath.split('/')[1]
    # 1.拿到实体-id、实体关系-id
    entityIdFileName = 'entity2id.txt'
    entity2id = dataForIdLoader(datasetPath, entityIdFileName)
    relationIdFileName = 'relation2id.txt'
    relation2id = dataForIdLoader(datasetPath, relationIdFileName)
    # 2.拿到训练的三元组
    trainFileName = 'train.txt'
    tripleList = dataForTripleLoader(datasetPath, trainFileName, entity2id, relation2id)
    return entity2id, relation2id, tripleList

def distanceL2(h,r,t):
    return np.sum(np.square(h + r - t))

def distanceL1(h,r,t):
    return np.sum(np.fabs(h + r - t))
    
class TransE:
    def __init__(self, entity2Id, relation2Id, tripleList,
                 embeddingDimensions=50, learningRate=0.01, margin=1, L1=True):
        self.entity2Id = entity2Id
        self.relation2Id = relation2Id
        self.tripleList = tripleList
        self.embeddingDimensions = embeddingDimensions
        self.learningRate = learningRate
        self.margin = margin
        self.L1 = L1
        # 用于存储向量的字典:{'id': [...](向量)}
        self.entityId2Vector = {}
        self.relationId2Vector = {}
        self.loss = 0

    # 将生成随机的数，表示实体、关系边的向量
    def emb_initialize(self):
        # 随机取上下限
        bottom = -6 / np.sqrt(self.embeddingDimensions)
        top = 6 / np.sqrt(self.embeddingDimensions)
        for entityId in self.entity2Id.values():
            tempVector = np.random.uniform(bottom, top, self.embeddingDimensions)
            self.entityId2Vector[entityId] = tempVector
        for relationId in self.relation2Id.values():
            tempVector = np.random.uniform(bottom, top, self.embeddingDimensions)
            tempVector = self.normalization(tempVector)
            self.relationId2Vector[relationId] = tempVector


    # epochs 训练轮数
    def train(self, epochs=100, batchTimes = 100):
        
        # 这里使用了整除，有可能最后的一些样本没有被抽出来
        batchSize = int(len(self.tripleList) / batchTimes)
        for epoch in range(epochs):
            startTime = time.time()
            self.loss = 0
            for entityId in self.entityId2Vector.keys():
                self.entityId2Vector[entityId] = self.normalization(self.entityId2Vector[entityId]);

            for i in range(batchTimes):
                if (i + 1) % 20 == 0:
                    text = "当前为第%d轮,总共有%d次,正在进行第%d次" % (epoch + 1, batchTimes, i + 1)
                    writeProgressText(text)
                # 1.随机抽取样本
                triples = random.sample(self.tripleList, batchSize)
                # 用来存放正负样本的列表
                sampleList = []
                # 2.生成负样本
                # 这里可以选择使用bern方法生成负样例
                for triple in triples:
                    corruptedTriple = copy.deepcopy(triple)
                    # 随机生成0~1
                    num = np.random.random(1)[0]

                    # 换头实体
                    if num > 0.5:
                        # corruptedTriple[0] = random.sample(self.entityId2Vector.keys(), 1)[0]
                        while corruptedTriple[0] == triple[0]:
                            corruptedTriple[0] = random.sample(self.entityId2Vector.keys(), 1)[0]
                    # 换尾实体
                    else:
                        # corruptedTriple[2] = random.sample(self.entityId2Vector.keys(), 1)[0]
                        while corruptedTriple[2] == triple[2]:
                            corruptedTriple[2] = random.sample(self.entityId2Vector.keys(), 1)[0]

                    if (triple, corruptedTriple) not in sampleList:
                        sampleList.append((triple, corruptedTriple))
                # 3.调整
                self.update_embeddings(sampleList)

            endTime = time.time()
            text = "总次数为:%d, 运行次数:%d, 耗时:%.2fs, loss值为:%d" % (epochs, epoch + 1, round(endTime - startTime, 2), self.loss)
            writeProgressText(text)
            
        text = "存储最终结果"
        writeProgressText(text)
        entityFileName = "%s_entity_%ddim_batch%d" % (datasetName, self.embeddingDimensions, batchTimes)
        relationFileName = "%s_relation_%ddim_batch%d" % (datasetName, self.embeddingDimensions, batchTimes)
        storeResult(entityFileName, self.entityId2Vector)
        storeResult(relationFileName, self.relationId2Vector)

    def update_embeddings(self, sampleList):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的elements都不同
        copy_entity = copy.deepcopy(self.entityId2Vector)
        copy_relation = copy.deepcopy(self.relationId2Vector)

        for correct_sample, corrupted_sample in sampleList:

            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[2]]
            relation_copy = copy_relation[correct_sample[1]]

            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[2]]

            correct_head = self.entityId2Vector[correct_sample[0]]
            correct_tail = self.entityId2Vector[correct_sample[2]]
            relation = self.relationId2Vector[correct_sample[1]]

            corrupted_head = self.entityId2Vector[corrupted_sample[0]]
            corrupted_tail = self.entityId2Vector[corrupted_sample[2]]

            # calculate the distance of the triples
            if self.L1:
                correct_distance = distanceL1(correct_head, relation, correct_tail)
                corrupted_distance = distanceL1(corrupted_head, relation, corrupted_tail)

            else:
                correct_distance = distanceL2(correct_head, relation, correct_tail)
                corrupted_distance = distanceL2(corrupted_head, relation, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            if loss > 0:
                self.loss += loss

                correct_gradient = 2 * (correct_head + relation - correct_tail)
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)

                if self.L1:
                    for i in range(len(correct_gradient)):
                        if correct_gradient[i] > 0:
                            correct_gradient[i] = 1
                        else:
                            correct_gradient[i] = -1

                        if corrupted_gradient[i] > 0:
                            corrupted_gradient[i] = 1
                        else:
                            corrupted_gradient[i] = -1

                correct_copy_head -= self.learningRate * correct_gradient
                relation_copy -= self.learningRate * correct_gradient
                correct_copy_tail -= -1 * self.learningRate * correct_gradient

                relation_copy -= -1 * self.learningRate * corrupted_gradient
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replaces the tail entity, the head entity's embedding need to be updated twice
                    correct_copy_head -= -1 * self.learningRate * corrupted_gradient
                    corrupted_copy_tail -= self.learningRate * corrupted_gradient
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replaces the head entity, the tail entity's embedding need to be updated twice
                    corrupted_copy_head -= -1 * self.learningRate * corrupted_gradient
                    correct_copy_tail -= self.learningRate * corrupted_gradient

                # normalising these new embedding vector, instead of normalising all the embedding together
                copy_entity[correct_sample[0]] = self.normalization(correct_copy_head)
                copy_entity[correct_sample[2]] = self.normalization(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity's embedding
                    copy_entity[corrupted_sample[2]] = self.normalization(corrupted_copy_tail)
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replace the head entity, update the head entity's embedding
                    copy_entity[corrupted_sample[0]] = self.normalization(corrupted_copy_head)
                # the paper mention that the relation's embedding don't need to be normalised
                copy_relation[correct_sample[1]] = relation_copy
                # copy_relation[correct_sample[1]] = self.normalization(relation_copy)

        self.entityId2Vector = copy_entity
        self.relationId2Vector = copy_relation

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    # 损失函数
    def hinge_loss(self,distCorrect,distCorrupt):
        return max(distCorrect - distCorrupt + self.margin, 0)
        

if __name__=='__main__':
    # 使用的数据集
    datasetPath = "./FB15k/"

    entity2Id, relation2Id, tripleList = dataLoader(datasetPath)
    clearProgressText()
    text = "实体数目：%d，关系数目：%d，训练三元组数目：%d" % (len(entity2Id), len(relation2Id), len(tripleList))
    writeProgressText(text)
    transE = TransE(entity2Id, relation2Id, tripleList, L1=False)
    transE.emb_initialize()
    transE.train(epochs=100)