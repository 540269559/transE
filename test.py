import numpy as np
import operator
import json
from transE import writeProgressText, clearProgressText, dataForIdLoader
import torch

# 测试的数据集
datasetName = "FB15k"
# 数据获取
# 获取训练好的实体和实体关系向量
def loadVectorData(fileName):
    vectorDict = {}
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            if len(arr) != 2:
                continue
            vectorDict[arr[0]] = json.loads(arr[1])
    return vectorDict

# 获取四个数据，实体-向量、关系边-向量、训练集、测试集
def loadTestTriple(fileName, entity2id, relation2id):
    testTriple = []
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.strip().split("\t")
            if len(arr) != 3:
                continue
            headId = entity2id[arr[0]]
            relationId = relation2id[arr[1]]
            tailId = entity2id[arr[2]]
            testTriple.append([headId, relationId, tailId])
    return testTriple

# 计算两个向量的距离
# 这里采用批量处理,计算距离,用到了torch
def distance(hVectorList,rVectorList,tVectorList, norm):
    head = torch.from_numpy(np.array(hVectorList))
    rel = torch.from_numpy(np.array(rVectorList))
    tail = torch.from_numpy(np.array(tVectorList))
    distance = head + rel - tail
    """
    input:输入的多维向量
    p:以第几范式作为输出值
    dim:按照第几维度进行范式计算，dim=0表示按列、dim=1表示按行进行范数的叠加
    """
    score = torch.norm(input=distance, p=norm, dim=1)
    return score.numpy()


class Test:
    def __init__(self, entity2vector, relation2vector, testTriple, isFit = True, norm = 1):
        
        self.entity2vector = entity2vector
        self.relation2vector = relation2vector
        self.testTriple = testTriple
        self.filter = isFit

        self.norm = norm

        self.hits10 = 0
        self.mean_rank = 0

        self.relation_hits10 = 0
        self.relation_mean_rank = 0

    # 实体准确率（rank）
    def entityRank(self):
        # 初始化数据
        hits = 0
        rank_sum = 0
        step = 1

        # 遍历测试集三元组
        for triple in self.testTriple:
            # 排序的头尾节点字典
            rankHeadDict = {}
            rankTailDict = {}
            # 头部替换
            headReplaceHeadEmbedding = []
            headReplaceRelationEmbedding = []
            headReplaceTailEmbedding = []
            # 存取头部替换的时候的三元组
            headTempTriple = []
            # 尾部替换
            tailReplaceHeadEmbedding = []
            tailReplaceRelationEmbedding = []
            tailReplaceTailEmbedding = []
            # 存取尾部替换的时候的三元组
            tailTempTriple = []

            # 依次尝试所有的头尾节点
            for entityId in self.entity2vector.keys():
                # 换头节点
                virtualHead = [entityId, triple[1], triple[2]]
                virtualTail = [triple[0], triple[1], entityId]
                # print("virtualHead", virtualHead)
                # 是否过滤，如果要过滤，则出现在test数据集的要过滤掉
                if not (self.filter and virtualHead in self.testTriple):
                    hVector = self.entity2vector[virtualHead[0]]
                    rVector = self.relation2vector[virtualHead[1]]
                    tVector = self.entity2vector[virtualHead[2]]
                    headReplaceHeadEmbedding.append(hVector)
                    headReplaceRelationEmbedding.append(rVector)
                    headReplaceTailEmbedding.append(tVector)
                    headTempTriple.append(tuple(virtualHead))

                if not (self.filter and virtualTail in self.testTriple):
                    hVector = self.entity2vector[virtualTail[0]]
                    rVector = self.relation2vector[virtualTail[1]]
                    tVector = self.entity2vector[virtualTail[2]]
                    tailReplaceHeadEmbedding.append(hVector)
                    tailReplaceRelationEmbedding.append(rVector)
                    tailReplaceTailEmbedding.append(tVector)
                    tailTempTriple.append(tuple(virtualTail))
            headReplaceDistance = distance(headReplaceHeadEmbedding, headReplaceRelationEmbedding, headReplaceTailEmbedding, self.norm)
            tailReplaceDistance = distance(tailReplaceHeadEmbedding, tailReplaceRelationEmbedding, tailReplaceTailEmbedding, self.norm)
            for i in range(len(headTempTriple)):
                rankHeadDict[headTempTriple[i]] = headReplaceDistance[i]
            for i in range(len(tailTempTriple)):
                rankTailDict[tailTempTriple[i]] = tailReplaceDistance[i]
            # 排序
            rankHeadDictSorted = sorted(rankHeadDict.items(), key=operator.itemgetter(1))
            rankTailDictSorted = sorted(rankTailDict.items(), key=operator.itemgetter(1))
            # 记录
            # 记录头节点
            # print(type(rankHeadDictSorted), rankHeadDictSorted[0])
            # return 
            for i in range(len(rankHeadDictSorted)):
                replaceHeadTriple = rankHeadDictSorted[i][0]
                if replaceHeadTriple[0] == triple[0]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            # 记录尾节点
            for i in range(len(rankTailDictSorted)):
                replaceTailTriple = rankTailDictSorted[i][0]
                if replaceTailTriple[2] == triple[2]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            step += 1
            if step % 100 == 0:
                text = "现在正在检测第%d个三元组,当前的hits@10为%.2f,当前的平均rank为%.2f" % (step, hits / (2 * step), rank_sum / (2 * step))
                writeProgressText(text, fileName='testStatus.txt')
        # 获得最终结果
        self.hits10 = hits / (2 * len(testTriple))
        self.mean_rank = rank_sum / (2 * len(testTriple))

    # 关系准确率（rank）
    def relationRank(self):
        hits = 0
        rank_sum = 0
        step = 1
        for triple in self.testTriple:
            rankDict = {}
            for relationId in self.relationDict.keys():
                virtualTriple = (triple[0], relationId, triple[2])
                if self.filter and virtualTriple in self.testTriple:
                    continue
                
                hVector = entityDict[virtualTriple[0]]
                rVector = entityDict[virtualTriple[1]]
                tVector = entityDict[virtualTriple[2]]
                rankDict[tuple(virtualTriple)] = distance(hVector, rVector, tVector)
            rankListSorted = sorted(rankDict.items(), key=operator.itemgetter(1))
            
            rank = 0
            for item in rankListSorted:
                if triple[1] == item[0][1]:
                    break
                rank += 1
            if rank < 10:
                hits += 1
            rank_sum += rank + 1

            step += 1
            if step % 100 == 0:
                print("relation step ", step, " ,hits ", hits, " ,rank_sum ", rank_sum)
        self.relation_hits10 = hits / len(testTriple)
        self.relation_mean_rank = rank_sum / len(testTriple)
    
if __name__ == '__main__':
    clearProgressText(fileName='testStatus.txt')
    # 1.拿到训练后的向量
    entity2id = dataForIdLoader('./FB15k/', 'entity2id.txt')
    relation2id = dataForIdLoader('./FB15k/', 'relation2id.txt')
    entity2vector = loadVectorData('./trainResult/FB15k_entity_50dim_batch100')
    relation2vector = loadVectorData('./trainResult/FB15k_relation_50dim_batch100')
    # 2.拿到训练三元组
    testTriple = loadTestTriple("./FB15k/test.txt", entity2id, relation2id)
    text = "数据加载完毕,实体数据:%d,实体关系数据:%d,测试三元组数据:%d" % (len(entity2vector), len(relation2vector), len(testTriple))
    writeProgressText(text, fileName='testStatus.txt')
    # 3.测试实体准确率
    test = Test(entity2vector, relation2vector, testTriple, isFit=False)
    test.entityRank()
    print("entity hits@10: ", test.hits10)
    print("entity meanrank: ", test.mean_rank)
    # """
    # Hits@10: 这是一个精确度（Precision）指标，用于评估在前10个最可能的结果中是否包含了正确的答案。在这段代码中，它被用来评估实体预测和关系预测的准确性。具体来说，对于每个测试三元组，如果正确的实体（或关系）在前10个排名中，则计数器会增加。最后，Hits@10 被计算为命中次数除以总的测试三元组数。
    # Mean Rank: 这是一个排名指标，用于衡量模型在所有测试三元组中正确答案的平均排名。排名越低表示模型性能越好。具体来说，对于每个测试三元组，算法会预测其正确的实体（或关系），然后计算其排名（在所有可能的实体或关系中）。最后，将所有测试三元组的排名加总，然后除以测试三元组的总数，得到平均排名。
    # """
    # # 4.测试关系准确率
    # test.relationRank()
    # print("relation hits@10: ", test.relation_hits10)
    # print("relation meanrank: ", test.relation_mean_rank)
    # # 5.保存文档
    # f = open("result.txt",'w')
    # f.write("entity hits@10: "+ str(test.hits10) + '\n')
    # f.write("entity meanrank: " + str(test.mean_rank) + '\n')
    # f.write("relation hits@10: " + str(test.relation_hits10) + '\n')
    # f.write("relation meanrank: " + str(test.relation_mean_rank) + '\n')
    # f.close()