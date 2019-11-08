# directedCorePeripheryPaper

This is the code for the paper "Core–Periphery Structure in Directed Networks"

It allows a replication of the results in the paper. 

Requirements:
  * Python3
  * Networkx
  * Numpy
  * Scipy
  * SciKit-Learn
  * graph-tool (only for the graphtool stochastic block model comparison)

The scoring file gives an interface to the each of the methods. We provide two different function for each of the methods, namely a standard method which produces a score and a method that ends in 'Class' which outputs a partition of the nodes.
Further, scoring also contains several additional variants of advanced hits, the method used in the paper is named, advancedHitsOrder.

Finally we note that the naming convention differs slightly from the convention in the paper as the code for our 1 parameter synthetic model the parameter in the generator is 0.5-p for p the parameter in the paper.

