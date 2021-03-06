import numpy
from scipy.stats import multivariate_normal

# Number of candidates in the election
num_candidates = 5

# Number of unordered pairs of distinct candidates, which is the dimension of the covariance matrix
num_pairs = num_candidates *(num_candidates -1)//2

# We need to define a pairing function between the unordered pairs of candidates and the integers 0,...,num_pairs-1, which we call codes.

# Store the vector mapping codes to pairs
pair_vector = [0]*num_pairs

# Populate the vector of pairs
k=0
for i in range(num_candidates):
  for j in range(i+1,num_candidates):
    pair_vector[k] = [i,j]
    k = k+1

# Turn a code into a pair
def depair(k):
  return pair_vector[k]

# There is an explicit function for turning a pair into a code
def pair(p):
  return p[1]-2*p[0]-1 + (num_candidates)*(num_candidates+1)//2 - (num_candidates-p[0])*(num_candidates-p[0]+1)//2

# This function defines the i,jth entry of the covariance matrix
def entries(i,j):
  x = depair(i)
  y = depair(j)
  if x[0] == y[0] and x[1] == y[1]:
    return 1
  if x[1] == y[0]:
    return -1/3
  if x[1] == y[1]:
    return 1/3
  if x[0] == y[0]:
    return 1/3
  if x[0] == y[1]:
    return -1/3
  return 0

# Populate the covariance matrix
cov = numpy.empty((num_pairs,num_pairs))
for i in range(num_pairs):
  for j in range(num_pairs):
    cov[i,j] = entries(i,j)

# random_var is a random variable with the multivariate normal distribution of margin graphs
# We can generate instances using random_var.rvs()
random_var = multivariate_normal(None,cov)

# random_var is not in a particularly nice form, as it is a 1-dimensional array for what should be two-dimensional data. 
# We generate, from an instance of the random variable, the margin graph
candidates = range(0,num_candidates)
def generate_margin_graph(rv):
    mg = [[-numpy.inf for _ in candidates] for _ in candidates]
    for c1 in candidates:
        for c2 in candidates:
            if (c1 < c2 and rv[pair([c1,c2])] > 0):
                mg[c1][c2] = rv[pair([c1,c2])]
            if (c1 > c2 and rv[pair([c2,c1])] < 0):
                mg[c1][c2] = -rv[pair([c2,c1])]
            if (c1 == c2):
                mg[c1][c2] = 0
    return mg

# random_margin_graph is a random variable with the multivariate normal distribution of margin graphs
random_margin_graph = generate_margin_graph(random_var.rvs())

# One can generate further random margin graphs by further calls to random_var.rvs()
# random_margin_graph[i][j] is the margin by which i defeats j, or infinity if j defeats i; and random_margin_graph[i][i] is 0
