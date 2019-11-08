from collections import Counter
import numpy as np


# tested
def getConnectionProb(G,groups,num_grp=4):
    cnts = __getCounts__(G,groups,num_grp) # max(groups)+1)
    nCats = Counter(groups)
    for i in range(4):
        for j in range(4):
            denom = float(nCats[i]*nCats[j])
            if cnts[i][j]!=0:
                cnts[i][j] = cnts[i][j]/denom
    return cnts

# tested
def getUpsets(G,groups,structType='directedCP'):
    if not isinstance(groups[0],(int,np.int32,np.int64)):
        print('I am assuming that this is a list of list of node ids')
        groups1 = [-1 for x in G]
        for g1 in range(len(groups)):
            for x in groups[g1]:
                groups1[x] = g1
        groups = groups1
    if structType=='directedCP':
        cs = [[0,1,0,0],[0,1,0,0],[0,1,1,1],[0,0,0,0]]
    elif structType=='bowTie':
        cs = []
        cs.append([1,0,1,0,0,0,0])
        cs.append([1,0,0,0,0,0,0])
        cs.append([0,0,0,0,0,0,0])
        cs.append([0,0,0,0,0,0,0])
        cs.append([0,0,0,0,0,0,0])
        cs.append([0,0,0,0,0,0,0])
        cs.append([0,0,0,0,0,0,0])
    else:
        raise Exception('Unknown structure ')
    cnts = __getCounts__(G,groups,len(cs))
    nCats = Counter(groups)
    offEdges = 0
    for i in range(len(cs)):
        for j in range(len(cs)):
            denom = float(nCats[i]*nCats[j])
            if cs[i][j]==0:
#                cnts[i][j] = [cnts[i][j], denom - cnts[i][j] ]
                cnts[i][j] = cnts[i][j]
                offEdges += cnts[i][j]
            else:
#                cnts[i][j] = [ denom - cnts[i][j],cnts[i][j] ]
                cnts[i][j] = denom - cnts[i][j]
    return cnts,offEdges


## tested
def __getCounts__(G,groups,num_gs):
    counts=[[0 for x in range(num_gs)] for y in range(num_gs)]
    for x in G:
        g1 = groups[x]
        for y in G[x]:
            g2 = groups[y]
            counts[g1][g2]+=1
    return counts
