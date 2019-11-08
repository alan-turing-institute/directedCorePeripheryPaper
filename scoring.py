import lowRankApproximation as lra
import numpy as np
import helperFunctions as hf
import networkx as nx
import maximisationApproach as ma
import time
import hitsAlg as ht
import hitsMk2 as hm
import hitsMk2NonVect as hmNV
import otherApproaches as oa
from collections import Counter
import itertools as it
from math import log
from fitStatistics import *
from fitStatistics import __getCounts__
import fitStatistics as fs

# Overall Method


def runAllClassMethods(G, exclude=['likelihoodMaximisationClass',
                                   'advancedHitsOldClass'], reorder=True,recordTime=False):
    result = {}
    methods = globals()
    names = list(methods.keys())
    print(names)
    for item in names:
        if item.endswith('Class') and item not in exclude:
            t1 = time.time()
            temp1 = methods[item](G, reorder=reorder)
            t2 = time.time()
            print([item, t2-t1])
            if recordTime:
                result[item] = temp1
            else:
                result[item] = [temp1,t2-t1]
    return result
#
#
#
#
# Helper Functions


# tested
def __convertScoresToClasses__(scores, G, reorder=True):
    scores = np.array(scores)
    # Test to short out the kmeans class if all nodes have score 0
    t1 = set(y for x in np.array(scores) for y in x)
    if len(t1) == 1:
        return [0, ]*scores.shape[0]
    t1 = hf.kmeansp2(scores, 4, 10000)
    if reorder:
        t2 = reorderConvertGroups(G, t1)
        return t2
    return t1


# tested
def __penalityHelperWrap__(L):
    Cin_score = (L.sum(axis=0, keepdims=True)).T
    Cout_score = (L.sum(axis=1, keepdims=True))
    return __penalityHelper__(L, Cin_score, Cout_score)


# tested
def __penalityHelper__(L, Cin_score, Cout_score):
        n = L.shape[0]
        assert(n == L.shape[1])
        temp1 = Cin_score + Cout_score
        temp2 = Cin_score - Cout_score
        Lt = L.T
        Pin_score = -Lt@temp2 - L@temp1
        Pout_score = L@temp2 - Lt@temp1
        for i in range(n):
            Pin_score[i] += L[i, i]*Cin_score[i]
            Pin_score[i] += L[i, i]*Cout_score[i]
            Pin_score[i] += L[i, i]*Cin_score[i]
            Pin_score[i] -= L[i, i]*Cout_score[i]
            Pout_score[i] -= L[i, i]*Cin_score[i]
            Pout_score[i] += L[i, i]*Cout_score[i]
            Pout_score[i] += L[i, i]*Cin_score[i]
            Pout_score[i] += L[i, i]*Cout_score[i]

        t1 = np.concatenate(
            [Cin_score, Cout_score, Pin_score, Pout_score],
            axis=1)
        return t1


# tested
def __reverseFormulationWrap__(L):
    Cin_score = (L.sum(axis=0, keepdims=True)).T
    Cout_score = (L.sum(axis=1, keepdims=True))
    return __reverseFormulation__(Cin_score, Cout_score)


# tested
def __reverseFormulation__(Cin, Cout):
        Pout = -Cout + Cout.max()
        Pin = -Cin + Cin.max()
        t1 = np.concatenate([Cin, Cout, Pin, Pout], axis=1)
        return t1


# tested
def __normHelper__(t1):
    t1 = np.array(t1)
    d1 = (t1*t1).sum(axis=1)
    d1 = (np.multiply(t1, t1)).sum(axis=1)
    d1 = np.sqrt(d1)
    d1[d1 == 0] = 1
    d1 = d1.reshape([t1.shape[0], 1])
    return np.divide(t1, d1)


# tested
def reorderConvertGroups(G, groups):

    if (len(groups) != len(G)):
        raise Exception('Number of nodes in graph and groups are not equal')

    counts = __getCounts__(G, groups, 4)
    currentBest = -np.inf

    numEdge = G.number_of_edges()
    numCat = Counter(groups)
    totDenom = len(G)**2
    for item in it.permutations(range(4)):

        ein = 0
        Tin = 0

        ein += counts[item[0]][item[1]]
        ein += counts[item[1]][item[1]]
        ein += counts[item[2]][item[1]]
        ein += counts[item[2]][item[2]]
        ein += counts[item[2]][item[3]]

        Tin += numCat[item[0]]*numCat[item[1]]
        Tin += numCat[item[1]]*numCat[item[1]]
        Tin += numCat[item[2]]*numCat[item[1]]
        Tin += numCat[item[2]]*numCat[item[2]]
        Tin += numCat[item[2]]*numCat[item[3]]

        eout = numEdge - ein
        Tout = totDenom - Tin
        l1 = 0
        if ein != 0:
            l1 += ein*log(ein/Tin)
        if ein != Tin:
            l1 += (Tin-ein)*log(1-ein/Tin)
        if eout != 0:
            l1 += eout*log(eout/Tout)
        if eout != Tout:
            l1 += (Tout-eout)*log(1-eout/Tout)
        if l1 > currentBest:
            currentBest = l1
            bestSol = item

    bestSol = {bestSol[x]: x for x in range(4)}

    return np.array([bestSol[x] for x in groups])
#
#
#
#
# Scoring method functions


# not tested
def degScores(G):
    data = [[G.in_degree(x) for x in range(len(G))], [G.out_degree(x) for x in
            range(len(G))]]
    data = list(zip(*data))
    return data


# not tested
def degClass(G, reorder=True):
    scores = degScores(G)
    return __convertScoresToClasses__(scores, G, reorder)


# not tested
def lowRankPenalityNoThres(G):
    L = lra.lowRankApproximation(G)
    t1 = __penalityHelperWrap__(L)
    return __normHelper__(t1)


# not tested
def lowRankPenalityNoThresClass(G, reorder=True):
    scores = lowRankPenalityNoThres(G)
    return __convertScoresToClasses__(scores, G, reorder)


# not tested
def lowRankPenality(G):
    L = lra.lowRankApproximationThreshold(G)
    t1 = __penalityHelperWrap__(L)
    return __normHelper__(t1)


# not tested
def lowRankPenalityClass(G, reorder=True):
    scores = lowRankPenality(G)
    return __convertScoresToClasses__(scores, G, reorder)


# not tested
def lowRankReverseNoThres(G):
    L = lra.lowRankApproximation(G)
    t1 = __reverseFormulationWrap__(L)
    return __normHelper__(t1)


# not tested
def lowRankReverseNoThresClass(G, reorder=True):
    scores = lowRankReverseNoThres(G)
    return __convertScoresToClasses__(scores, G, reorder)


# not tested
def lowRankReverse(G):
    L = lra.lowRankApproximationThreshold(G)
    t1 = __reverseFormulationWrap__(L)
    return __normHelper__(t1)


# not tested
def lowRankReverseClass(G, reorder=True):
    scores = lowRankReverse(G)
    return __convertScoresToClasses__(scores, G, reorder)


# not tested
def __getHits__(G):
    hubs, authorities = ht.HITS_alg(G)
    hubs = np.array([[hubs[x] for x in range(len(G))]]).T
    authorities = np.array([[authorities[x] for x in range(len(G))]]).T
    return hubs, authorities


def hitsPenality(G):
    Cin_score, Cout_score = __getHits__(G)
    A = nx.to_scipy_sparse_matrix(G, sorted(G.nodes()))
    t1 = __penalityHelper__(A, Cin_score, Cout_score)
    return __normHelper__(t1)


def hitsPenalityClass(G, reorder=True):
    scores = hitsPenality(G)
    return __convertScoresToClasses__(scores, G, reorder)


def hitsReverse(G):
    Cin_score, Cout_score = __getHits__(G)
    t1 = __reverseFormulation__(Cin_score, Cout_score)
    return __normHelper__(t1)


def hitsReverseClass(G, reorder=True):
    scores = hitsReverse(G)
    return __convertScoresToClasses__(scores, G, reorder)


def lowRankRaw(G, dim=2):
    uLwDm, sLwDm, vhLwDm = lra.lowRankApproximation(G, dim, False)
    t1 = [uLwDm, vhLwDm.T]
    t1 = np.concatenate(t1, axis=1)
    return __normHelper__(t1)


def lowRankRawClass(G, reorder=True):
    scores = lowRankRaw(G)
    t1 = __normHelper__(scores)
    return __convertScoresToClasses__(t1, G, reorder)


def lowRankRawScale(G, dim=2):
    uLwDm, sLwDm, vhLwDm = lra.lowRankApproximation(G, dim, False)
    t1 = [uLwDm@np.sqrt(sLwDm), vhLwDm.T@np.sqrt(sLwDm)]
    t1 = np.concatenate(t1, axis=1)
    return __normHelper__(t1)


def lowRankRawScaleClass(G, reorder=True):
    scores = lowRankRawScale(G)
    t1 = __normHelper__(scores)
    return __convertScoresToClasses__(t1, G, reorder)


def hillClimbApproach(G, numberOfAttempts=10):
    t1 = ma.hillClimbApproach(G, numberOfAttempts)
    return __convertClassesToScore__(t1)


def hillClimbApproachClass(G, numberOfAttempts=10, reorder=True):
    return ma.hillClimbApproach(G, numberOfAttempts)


def __convertClassesToScore__(t1):
    result = np.zeros([len(t1), 4])
    for i in range(len(t1)):
        result[i, t1[i]] = 1


def likelihoodMaximisation(G, numberOfAttempts=10):
    t1 = ma.likelihoodMaximisation(G, numberOfAttempts)
    return __convertClassesToScore__(t1, G)


def likelihoodMaximisationClass(G, numberOfAttempts=10, reorder=True):
    return ma.likelihoodMaximisation(G, numberOfAttempts)


def advHitsVectPlusOneTime(G):
    return hm.advancedHitsVectFullPlus1AtTime(G)


def advHitsVectPlusOneTimeClass(G, reorder=True):
    scores = advHitsVectPlusOneTime(G)
    return __convertScoresToClasses__(scores, G, reorder)


def advHitsVectPlusOneTimeGroup(G):
    return hm.advancedHitsVect_Group_FullPlus1AtTime(G)


def advHitsVectPlusOneTimeGroupClass(G, reorder=True):
    scores = advHitsVectPlusOneTimeGroup(G)
    return __convertScoresToClasses__(scores, G, reorder)


def advancedHitsOrder(G):
    return hm.advancedHitsOrder(G)


def advancedHitsOrderClass(G, reorder=True):
    scores = advancedHitsOrder(G)
    return __convertScoresToClasses__(scores, G, reorder)


def advancedHitsOrderGroup(G):
    return hm.advancedHitsOrderGroup(G)


def advancedHitsOrderGroupClass(G, reorder=True):
    scores = advancedHitsOrderGroup(G)
    return __convertScoresToClasses__(scores, G, reorder)


def advHitsVect(G):
    return hm.advancedHitsVect(G)


def advHitsVectClass(G, reorder=True):
    scores = advHitsVect(G)
    return __convertScoresToClasses__(scores, G, reorder)


def advancedHits(G):
    return hmNV.advancedHits(G)


def advancedHitsClass(G, reorder=True):
    scores = advancedHits(G)
    return __convertScoresToClasses__(scores, G, reorder)




try:
    import graphToolHelper as gth

    def graphToolClass(G, reorder=True):
        return gth.getGroupsNetworkx(G)

    def graphToolDegCorrClass(G, reorder=True):
        return gth.getGroupsNetworkxDegreeCorrect(G)
except:
    print('problem with graphtool implementation')


def SaPa1Class(G, reorder=True):
    return oa.run_SaPa(G, 1, 4)


def SaPa2Class(G, reorder=True):
    return oa.run_SaPa(G, 2, 4)


def disum1Class(G, reorder=True):
    return oa.disum_reg_MC(G, 4)[0]


def disum2Class(G, reorder=True):
    return oa.disum_reg_MC(G, 4)[1]


def disum3Class(G, reorder=True):
    return oa.disum_reg_MC(G, 4)[2]


def disumCombined3Class(G, reorder=True):
    t1 = oa.disum_reg_MC(G, 2)
    t2 = t1[0]*2 + t1[1]
    return t2


getUpsets = fs.getUpsets
