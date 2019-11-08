from sklearn.cluster import KMeans


# Kmeans code - simple wrapper around the sklearn version with the same
# interface as the R version
#
# Function for kmeans++
# kmeansp2 <- function(x, k, iter.max = 10, nstart = 1, ...) {
#  n <- nrow(x) # number of data points
#  centers <- 0 # IDs of centers
#  # distances[i, j]: The distance between x[i,] and x[centers[j],]
#  distances <- matrix(numeric(n * (k - 1)), ncol = k - 1)
#  # the best result among <nstart> iterations
#  res.best <- list(tot.withinss = Inf)
#
#  for (rep in 1:nstart) {
#    pr <- rep(1, n) # probability for sampling centers
#    for (i in 1:(k - 1)) {
#      centers[i] <- sample.int(n, 1, prob = pr)# Pick up the ith center
#      # Compute (the square of) distances to the center
#      distances[, i] <- colSums((t(x) - x[centers[i], ])^2)
#      # Compute probaiblity for the next sampling
#      pr <- distances[cbind(1:n, max.col(-distances[, 1:i, drop = FALSE]))]
#    }
#    centers[k] <- sample.int(n, 1, prob = pr)
#  }
#  ## Perform k-means with the obtained centers
#  res <- kmeans(x, x[centers, ], iter.max = iter.max, nstart = 1, ...)
#  res$inicial.centers <- x[centers, ]
#  ## Store the best result
#  if (res$tot.withinss < res.best$tot.withinss) {
#    res.best <- res
#  }
#  res.best
# }

def kmeansp2(x, k, iterMax=None):
    if iterMax is None:
            iterMax = x.shape[0]*k
    if iterMax < 50:
            print('iterMax is quite low')
    clusterer = KMeans(n_clusters=k, max_iter=iterMax)
    return clusterer.fit_predict(x)
