import networkx as nx
import numpy as np
import random as rd
from collections import Counter
from math import log
import time


# tested
def __considerMovement__(G, currentSolution, node, savedVars):  # Basic test
    # This function will give 4 scores indicating the relative score of each of
    # the groups. Note, that this is not necessarily the actual likelihood, but
    # we guarentee that the differences are preserved.
    n = len(G)
    cs = savedVars['cs']
    numCats = savedVars['numCats']
    curGroup = currentSolution[node]
    numCats[curGroup] -= 1
    selfLoop = node in G[node]
    result = []

    forward, backward = savedVars['forwardBackward'][node]
    ein = savedVars['ein']

    for j in range(4):
        ein -= cs[curGroup][j]*forward[j]
        ein -= cs[j][curGroup]*backward[j]
    if selfLoop:
        if cs[curGroup][curGroup]:
            ein -= 1

    einOld = ein
    for i in range(4):
        ein = einOld
        for j in range(4):
            ein += cs[i][j]*forward[j]
            ein += cs[j][i]*backward[j]
        if selfLoop:
            ein += cs[i][i]
        numCats[i] += 1
        eout = savedVars['numEdges']-ein
        Tin = (numCats[0]+numCats[1]+numCats[2])*numCats[1]
        Tin += numCats[2]*(numCats[2]+numCats[3])
        Tout = n*n-Tin
        l1 = 0
        if ein != 0:
            l1 += ein*log(ein/Tin)
        if ein != Tin:
            l1 += (Tin-ein)*log(1-ein/Tin)
        if eout != 0:
            l1 += eout*log(eout/Tout)
        if eout != Tout:
            l1 += (Tout-eout)*log(1-eout/Tout)
        result.append(l1)
        numCats[i] -= 1
    numCats[curGroup] += 1
    return result


# tested
def getEin(G, coms):
    cs = __getCorrectStructure__()
    ein = 0
    for x in G:
        if coms[x] != 3:
            for y in G[x]:
                if cs[coms[x]][coms[y]]:
                    ein += 1
    return ein


# tested
def __updateVarsOnComChange__(savedVars, G, currentSolution, node, newCom):
    oldGroup = currentSolution[node]
    currentSolution[node] = newCom
    savedVars['numCats'][oldGroup] -= 1
    savedVars['numCats'][newCom] += 1
    cs = savedVars['cs']
    for x in G[node]:
        if x != node:
            savedVars['forwardBackward'][x][1][oldGroup] -= 1
            savedVars['forwardBackward'][x][1][newCom] += 1
    for x1 in G.in_edges(node):
        x = x1[0]
        if x != node:
            savedVars['forwardBackward'][x][0][oldGroup] -= 1
            savedVars['forwardBackward'][x][0][newCom] += 1
    coms = currentSolution
    for y in G[node]:
        if node == y:
            continue
        savedVars['ein'] -= cs[oldGroup][coms[y]]
        savedVars['ein'] += cs[newCom][coms[y]]
    for y1 in G.in_edges(node):
        y = y1[0]
        if node == y:
            continue
        savedVars['ein'] -= cs[coms[y]][oldGroup]
        savedVars['ein'] += cs[coms[y]][newCom]
    if node in G[node]:
        savedVars['ein'] -= cs[oldGroup][oldGroup]
        savedVars['ein'] += cs[newCom][newCom]


# tested
def getForwardBackward(G, node, coms):
    # Get groups that each node connects to
    forward = [0, 0, 0, 0]
    backward = [0, 0, 0, 0]
    for x in G[node]:
        if x != node:
            forward[coms[x]] += 1
    for x1 in G.in_edges(node):
        x = x1[0]
        if x != node:
            backward[coms[x]] += 1
    return forward, backward


# tested
def getForwardBackwardAll(G, coms):
    # Get groups that each node connects to
    result = []
    for i in range(len(G)):
        result.append([[0,0,0,0],[0,0,0,0]])
    for i in G:
        for j in G[i]:
            if i!=j:
                result[i][0][coms[j]]+=1
                result[j][1][coms[i]]+=1
    return result


def __likelihoodMaximisation__(G, initialComs):
    # We are going to use the procedure that is described in Newmans paper.
    # saved some variables to speed up the computation
    change = True
    likelihoodTest = __likelihood__(G, initialComs)
    maxSol = initialComs[:]
    curMax = likelihoodTest
    # main loop
    maxLikeHelper = __likelihoodMaximisationHelper__
    while change:
        toConsider = set(range(len(G)))
        maxSol, curMax, change = maxLikeHelper(G, maxSol, curMax, toConsider)
    return maxSol, curMax


# Basic tests
def __likelihoodMaximisationHelper__(G, maxSolution, currentMax, toConsider):
    change = False
    currentSol = maxSolution[:]
    currentLike = currentMax
    # Need to update the saved variables here as we have chosen a point in
    # the previous sequence to update to
    savedVars = __constructSavedVars__(G, currentSol)
    # Move each node once
    while len(toConsider) > 0:
        bestMove = -np.inf
        bestNode = None
        bestCom = None

        for node in toConsider:
            # Look at the moves for this node
            solution = __considerMovement__(G, currentSol, node, savedVars)

            # discover the best community
            sortedSol = sorted(list(zip(solution, range(4))))
            if sortedSol[-1][1] != currentSol[node]:
                newCom = sortedSol[-1][1]
            else:
                newCom = sortedSol[-2][1]
            diff = solution[newCom]-solution[currentSol[node]]
            if diff > bestMove:
                bestMove = diff
                bestNode = node
                bestCom = newCom

        # update the saved variables
        toConsider.remove(bestNode)

        # update the saved variables with the change
        __updateVarsOnComChange__(savedVars, G, currentSol, bestNode, bestCom)

        # store the best structure
        currentLike += bestMove
        if currentLike > currentMax:
            currentMax = currentLike
            maxSolution = currentSol[:]
            change = True
    return maxSolution, currentMax, change


# Tested
def __likelihood__(G, coms):
    n = len(G)
    numCats = Counter(coms)
    ein = 0
    eout = 0
    ein = getEin(G, coms)
    eout = G.number_of_edges()-ein
    Tin = (numCats[0]+numCats[1]+numCats[2])*numCats[1]
    Tin += numCats[2]*(numCats[2]+numCats[3])
    Tout = n*n-Tin
    l1 = 0
    if ein != 0:
        l1 += ein*log(ein/Tin)
    if ein != Tin:
        l1 += (Tin-ein)*log(1-ein/Tin)
    if eout != 0:
        l1 += eout*log(eout/Tout)
    if eout != Tout:
        l1 += (Tout-eout)*log(1-eout/Tout)
    return l1


# Not tested (simple wrapper arround networkx function)
def __convertToIntegers__(G):
    nodes = sorted(G.nodes())
    G1 = nx.convert_node_labels_to_integers(G, ordering='sorted')
    return G1, nodes


# Not tested (simple routine)
def likelihoodMaximisation(G1, numberOfAttempts=10):
    G, mapping = __convertToIntegers__(G1)
    currentMax = -np.inf
    currentSolution = None
    for x in range(numberOfAttempts):
        initialComs = [rd.randint(0, 3) for x in range(len(G1))]
        coms, likelihood = __likelihoodMaximisation__(G, initialComs)
        # Compute likelihood independently as a test
        likelihoodTest = __likelihood__(G, coms)
        assert(abs(likelihood-likelihoodTest) < 10**(-8))
        if likelihoodTest > currentMax:
            currentMax = likelihoodTest
            currentSolution = coms
    return currentSolution


# tested
def __estimatePandQ__(G, cats, savedVars):
    p = 0
    q = 0
    numCats = savedVars['numCats']
    n = len(G)
    p = savedVars['ein']
    q = savedVars['numEdges']-p
    pDenom = (numCats[0]+numCats[1]+numCats[2])*numCats[1]
    pDenom += numCats[2]*(numCats[2]+numCats[3])
    qDenom = n*n-pDenom
    p = p/pDenom
    q = q/qDenom
    return p, q


# does not need a test
def __getCorrectStructure__():
    correctStructure = []
    correctStructure.append([0, 1, 0, 0])
    correctStructure.append([0, 1, 0, 0])
    correctStructure.append([0, 1, 1, 1])
    correctStructure.append([0, 0, 0, 0])
    return correctStructure


# No need to test
def hillClimbApproach(G1, reps=10):
    # convert graph to deal with non integer graphs
    G, mapping = __convertToIntegers__(G1)
    curBest = -np.inf
    for x in range(reps):
        result = __hillClimbApproachHelper__(G)
        l1 = __likelihood__(G, result)
        if l1 > curBest:
            curBest = l1
            curPart = result
    return curPart
    return list(zip(mapping, curPart))


# No test (wrapper around other routines)
def __constructSavedVars__(G, coms):
    savedVars = {}
    savedVars['numEdges'] = G.number_of_edges()
    savedVars['numCats'] = Counter(coms)
    savedVars['ein'] = getEin(G, coms)
    savedVars['cs'] = __getCorrectStructure__()
    # Make the initial forward and backward arrays
    savedVars['forwardBackward']=getForwardBackwardAll(G,coms)
#    savedVars['forwardBackward']={}
#    for node in range(len(G)):
#        savedVars['forwardBackward'][node] = getForwardBackward(G, node, coms)
    return savedVars


# No test just a wrapper
def __hillClimbApproachHelper__(G):
    n = len(G)

    # initial guess on the groups
    coms = [rd.randint(0, 3) for i in range(n)]

    # save some basic data
    savedVars = __constructSavedVars__(G, coms)
    order = list(range(n))

    for runIn in range(5000):
        # update p and q
        p, q = __estimatePandQ__(G, coms, savedVars)

        # so that the algorithm doesnt get stuck someone silly
        if p < q:
            p, q = q, p

        numChanges = 0
        rd.shuffle(order)
        for node in order:
            scores = __considerMovement__(G,  coms,  node,  savedVars)
            newCom = max(list(zip(scores, range(4))))[1]
            if coms[node] != newCom:
                numChanges += 1
                __updateVarsOnComChange__(savedVars, G, coms, node, newCom)
                assert(coms[node] == newCom)
        if numChanges == 0:
            break
    if numChanges>0:
        print('Warning this hill climb replicate did not converge')
    return coms
