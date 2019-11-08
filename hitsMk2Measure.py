
def getScaling(ns,k,reps=10):
    result1=[]
    result2=[]
    for n in ns:
        timings1 = []
        timings2 = []
        for rep in range(reps):
            G = nx.erdos_renyi_graph(n,k,directed=True)
            t1=time.time()
            runInOrderSpec(G,maxiter=1000)
            t2=time.time()
            print(['advH',rep,t2-t1])
            timings1.append(t2-t1)
            t1=time.time()
            sc.hillClimbApproach(G)
            t2=time.time()
            print(['hill',rep,t2-t1])
            timings2.append(t2-t1)
        result1.append(timings1)
        result2.append(timings2)
    return result1,result2


def getTimings(G,reps=10):
    timings = []
    for rep in range(reps):
        t1=time.time()
        runInOrderSpec(G,maxiter=1000)
        t2=time.time()
        print([rep,t2-t1])
        timings.append(t2-t1)
    print(['min',sum(timings)/float(len(timings))])
    print(['min',min(timings)])
    print(['max',max(timings)])
    print(['num>1',sum(x>1 for x in timings)])
    print(['num>5',sum(x>5 for x in timings)])
    return timings
