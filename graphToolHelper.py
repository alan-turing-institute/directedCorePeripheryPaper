import graph_tool as gt
from graph_tool import inference 

def __networkxToGraphTool__(G):
    g_gt = gt.Graph()
    nodeMap = {}
    vertices = g_gt.add_vertex(len(G))
    for item in zip(vertices,sorted(G)):
        v1,x = item
        nodeMap[x]=v1
    edgeList = []
    for x in G:
        x_gt=nodeMap[x]
        for y in G[x]:
            edgeList.append((x_gt,nodeMap[y]))
    g_gt.add_edge_list(edgeList)
#            g_gt.add_edge(x_gt,nodeMap[y])
    return g_gt,nodeMap


def getGroupsNetworkxDegreeCorrect(G):
    g_gt,nodeMap = __networkxToGraphTool__(G)
    result = inference.minimize.minimize_blockmodel_dl(g_gt,B_min=4,B_max=4,deg_corr=True)
    blocks = result.get_blocks()
    finalResult=[]
    for item in nodeMap:
        finalResult.append(blocks[nodeMap[item]])
    return finalResult 

def getGroupsNetworkx(G):
    g_gt,nodeMap = __networkxToGraphTool__(G)
    result = inference.minimize.minimize_blockmodel_dl(g_gt,B_min=4,B_max=4,deg_corr=False)
    blocks = result.get_blocks()
    finalResult=[]
    for item in nodeMap:
        finalResult.append(blocks[nodeMap[item]])
    return finalResult 
