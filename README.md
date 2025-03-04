# numerical_algorithms

Some math functions which I found interesting to implement. Mostly about Calculus and Optimization.

## double_integral

As the name suggests, this function is made to compute and visualize double integrals (with non-rectangular domains).
It does not pretend to efficiency or brevity. The main purpose was to create a simple self-explainable code
and a clear visualization.

Example:
```python
double_integral(lambda x, y: 1 / (3 * x + 2 * y), ['x < y < 5 * x', '2 < x + 2 * y < 4'],
                plot_on=True, precision=300)
```
```
   Precision = 300    |                Boundaries:                
-------------------------------------------------------------------
   Limits received:   |   X =    |  X type  |   Y =    |  Y type  
-------------------------------------------------------------------
1)        x < y       |    y     | function |    x     | function 
2)        y < 5*x     |   y/5    | function |   5*x    | function 
3)        2 < x+2*y   | 2 - 2*y  | function | 1 - x/2  | function 
4)    x+2*y < 4       | 4 - 2*y  | function | 2 - x/2  | function 
-------------------------------------------------------------------
Vertices of the domain are: [(4/3, 4/3), (2/11, 10/11), (2/3, 2/3), (4/11, 20/11)]
Computing the integral... : 100%|#############| 93637/93637 [00:02<00:00, 40238.72 points checked/s]
            Integral value:  0.172032
                       Min:  0.171623
                       Max:  0.172442
```
![output](https://github.com/user-attachments/assets/85c1d459-57a0-401c-b13d-19cd8e25b7f4)

More datailed explanation is provided within the function docstrings.

## monte_carlo_integration

Straight-forward implementation of MCI. The goal is to demonstrate all the steps of MCI
from taking an arbitrary density function, validating it over the domain, sampling points,
finding MC estimator and comparing to the Newton's-Côtes result.

Visualization and animation are also possible.

Example
```python
def f(x):
  return np.sin(x) + np.cos(x)

def pdf(x):
  return 0.5 * x + 1 if x < 1 else 2.5 - x

mci(f, pdf, [(0, 2)], n_samples=1000, plot=True, verbose=True)
```
```
Task:
Integrand:                      np.sin(x) + np.cos(x)
Raw PDF:                       0.5 * x + 1 if x < 1 else 2.5 - x
PDF validation required:      True
Sampling function provided:   Not Provided
Number of variables:          1
Boundaries of integration:    [0, 2]
Infinite domain:              False
Number of samples:            1000
Plotting option:              True
Animation:                    False

Checking PDF:
PDF minimum:                  0.5001605215288005
PDF integrates to:            2.25
  it will be multiplied by:   0.444

Rejection Sampling from PDF:
Proposal (q(x)):              uniform
pdf(x) / q(x) upper bound:    0.67
Collecting draws from PDF: 100%|██████████| 1000/1000 [00:00<00:00, 239976.20it/s]
Acceptance rate:              0.75
Ranges:                       ['(8.62e-05, 2)']
Mean(s):                      [0.90408331]
Variance(s):                  [0.2872578]

Monte Carlo estimation:
Computing MC estimator: 100%|██████████| 1000/1000 [00:00<00:00, 412378.72it/s]
Number of omitted samples:    0
Variance:                     4.1791e-05
SD:                           0.0064646
The final estimation:         2.33
Confidence interval:          (2.3136 ... 2.3389) p=0.05
Reference value               2.3254
Error:                        0.00080659
```
![3](https://github.com/user-attachments/assets/044db8f1-5067-4d10-82fb-9dccc083a6c2)



## unconstrained_optimization

Attempt to implement the most basic optimization algorithms. The implementation is completely
based on the famous textbook:
```
Nocedal, J., & Wright, S. J. (2006). Numerical optimization. In Springer Series in Operations
Research and Financial Engineering (pp. 1-664). (Springer Series in Operations
Research and Financial Engineering). Springer Nature.
```

Of course, there exist innumerable tools (including scipy state-of-art optimize function). The
goal of this code is to trace step-by-step the process of unconstrained optimization,
write down logs and intermediate computations. This approach results into fairly slow performance.

At first a Task instance is created using a Callable or a string function, x0 - initial point (optional),
x_min - true minimizers (optional), n_var - number of variables (optional), external_grad - gradient
function (optional), external_hess - hessian function (optional).
Optional arguments are replaced with automatically as much as possible.

Secondly, an optimizer instance is created like Newton(), ConjugateGradient(), etc.
Passing Task into Optimizer().optimize(Task) returns the found soltion and a dataframe of logs.

Created Task() or loaded Optimizer() have their custom representations (see example).

Algorithms implemented:

Step Search:
- fixed step lengtg
- backtracking
- weak Wolfe conditions
- strong Wolfe conditions
- automatic step (based on previous step)

Optimizers:

- Steepest Decent

- Newton method
  - Newton method with Hessian modification

- QuasiNewton
  - QuasiNewton method SR1
  - QuasiNewton method DFP
  - QuasiNewton method BFGS
  - QuasiNewton method SR1 wth trust region
 
- Conjugate Gradient
  - Conjugate Gradient Fletcher-Reeves (FR)
  - Conjugate Gradient Polak-Ribiere (PR)
  - Conjugate Gradient Hestenes-Stiefel (HS)
  - Conjugate Gradient Dai-Yuan (DY)
  - Conjugate Gradient (linear task)

Additional:
- 1D visualization
- 2D visualization

Example

```python
task = Task('100 * (x_1 - x_2^2)^2 + (1 - x_2)^2', x0=[2, 0])
print(task)

Newton_opt = Newton()
QN_opt = QuasiNewton()
CG_opt = ConjugateGradient()

Newton_opt.optimize(task, modify_hessian=True)
print(Newton_opt)

QN_opt.optimize(task, algorithm='DFP')
print(QN_opt)

CG_opt.optimize(task, algorithm='PR')
print(CG_opt)
CG_opt.visualize2d()
```

```output
----------------- TASK -----------------
Function (string):            100 * (x[0] - x[1]**2)**2 + (1 - x[1])**2
Number of variables:          2
Initial point:                [2. 0.]
True optimizers:              not provided
Gradient:                     automatically
Hessian:                      automatically
----------------------------------------
------------- OPTIMIZATION -------------
Function                           100 * (x[0] - x[1]**2)**2 + (1 - x[1])**2
Method                             :Newton
Stop condition                     :Module of gradient = 1e-06
Iter limit                         :100000
Step length search                 :Wolfe
   Mode                            :strong
   Initial length                  :1
   Constant c1                     :0.1
   Constant c2                     :0.9
Starting point                     :[2. 0.]
Modify Hessian                     :True
Status                             :Solved: ['1', '1']
Runtime                            :0.8315s
Iterations                         :8
Last gradient module               :1.118e-08
----------------------------------------
------------- OPTIMIZATION -------------
Function                           100 * (x[0] - x[1]**2)**2 + (1 - x[1])**2
Method                             :Quasi Newton
Stop condition                     :Module of gradient = 1e-06
Iter limit                         :10000
Step length search                 :Wolfe
   Mode                            :strong
   Initial length                  :1
   Constant c1                     :0.1
   Constant c2                     :0.9
Starting point                     :[2. 0.]
Algorithm                          :DFP
Status                             :Solved: ['1', '1']
Runtime                            :1.037s
Iterations                         :11
Last gradient module               :8.03e-08
----------------------------------------
------------- OPTIMIZATION -------------
Function                           100 * (x[0] - x[1]**2)**2 + (1 - x[1])**2
Method                             :Conjugate Gradient
Stop condition                     :Module of gradient = 1e-06
Iter limit                         :10000
Step length search                 :Wolfe
   Mode                            :strong
   Initial length                  :1
   Constant c1                     :0.1
   Constant c2                     :0.9
Starting point                     :[2. 0.]
Algorithm                          :PR
Status                             :Solved: ['1', '1']
Runtime                            :1.333s
Iterations                         :16
Last gradient module               :2.854e-08
----------------------------------------
```
![opt](https://github.com/user-attachments/assets/eef5b5ec-2606-4855-a064-e8d693d2f77f)



## community_analysis

It is a tool which takes a graph as its adjacency matrix (plus optionally a dictionary of node names - if not, indices of adjacency matrix will be also the node names).

After a Graph() instance is created, a lot of community analysis metrics and properties (both for the graph and individual nodes) can be extracted via the following methods

  1. shortest_paths:
        Searching for paths from node "_from" to the node "_to". (all shortest path are returned)
        If there is no path: False is returned.

  2. distance:
        Returns the cost of the shortest path between 2 nodes.

  3. is_complete:
        Checks if every node is connected to every other node directly.

  4. is_connected
        Checks if every node is reachable from any other node.

  5. is_regular:
        Check if every node has the same number of adjacent edges.

  6. is_weighted:
        Checks if edges have different weights.

  7. is_directed:
        Checks if the adjacent matrix is symmetric.

  8. is_moore:
        Checks if the graph is Moore Bound.

  9. is_acyclic:
        Check if there are no cycles in the graph.

  10. is_tree:
        Checks if the graph can be called a tree (connected and acyclic).

   11. diameter:
        Returns the maximum shortest path.

  12. eccentricity:
        Returns the maximum distance from the given vertex to other vertices.

  13. radius:
        Returns the minimum eccentricity of all vertices.

  14. average_distance_of_vertex:
        Returns the average distance from the given vertex to every other vertex.

  15. average_distance_of_graph:
        Returns the mean of average distances of all vertices.

  16. stress_centrality:
        Returns the total number of shortest paths going through the given node.

  17. betweenness_centrality:
        Returns the sum of fractions of shortest paths between each pair of nodes (not containing the given one), which goes through the given node.

  18. degree_centrality:
        Returns the degree of a node.

  19. k_hop_centrality:
        Returns the number of nodes, which are reachable through k edges from the given one. (can be normalized)

  20. k_rank_centrality:
        Returns the rank of the given node among its own k-hop neighbours sorted according to their degrees.

  21. closeness_centrality:
        Returns 1 / sum of distances from the given node to every other node.

  22. clustering_coefficient_of_vertex:
        Returns the fraction of existing edges between the neighbours of the given node to the theoretically
        maximum number of edges.

  23. clustering_coefficient_of_graph:
        Returns the mean of individual clustering coefficients of all vertices.

  24. neighbourhood_overlap:
        Returns the fraction of common neighbours to all the neighbours of the 2 given nodes.

  25. minimum_spanning_tree:
        Return the adjacency matrix of a minimum spanning tree constructed according to Prim-Jarnik or Kruskal's
        algorithm.

  26. summary:
        Returns the report with main measures of the graph (not parametrizable).


Methods describing individual nodes can usually be used with no node provided: in such a case a dictionary of individual metric for every node is returned.

'summary' methods provided an overview of what information can be potentially extracted from Graph().

Notice: the implementation is designed to be self-explainable and verbose. The efficiency was not a priority, so that the tool is made for rather compact, manually checkable graphs.

Example:
![graph](https://github.com/user-attachments/assets/9ca4f53e-48f2-467f-8023-5499d61c9150)

```python
matrix_1 = np.array(
        [  # A, B, C, D, E, F, G
            [0, 1, 1, 1, 0, 0, 1], # A
            [1, 0, 1, 0, 0, 0, 0], # B
            [1, 1, 0, 1, 0, 0, 0], # C
            [1, 0, 1, 0, 1, 1, 1], # D
            [0, 0, 0, 1, 0, 0, 0], # E
            [0, 0, 0, 1, 0, 0, 0], # F
            [1, 0, 0, 1, 0, 0, 0]  # G
        ]
    )

names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

g = Graph(matrix_1, names)
print(g.summary())
```

```
This graph:
	 - is connected
	 - is NOT regular
	 - is NOT directed
	 - is NOT weighted
	 - is NOT complete
	 - is NOT acyclic
	 - is NOT a Moore graph
	 - is NOT a Tree

Main Measures:
	 Diameter (geodesic) =   3
	 Diameter (not geodesic) =   3
	 Radius (geodesic) =   2
	 Radius (not geodesic) =   2
	 Average distance (geodesic) = 1.66667
	 Average distance (not geodesic) = 1.66667
	 Clustering coefficient = 0.480952

Minimum Spanning Trees:
by Prim-Jarnik:
[[0 0 0 0 0 0 1]
 [0 0 1 0 0 0 0]
 [0 1 0 1 0 0 0]
 [0 0 1 0 1 1 1]
 [0 0 0 1 0 0 0]
 [0 0 0 1 0 0 0]
 [1 0 0 1 0 0 0]]

by Kruskal:
[[0 0 0 0 0 0 1]
 [0 0 1 0 0 0 0]
 [0 1 0 1 0 0 0]
 [0 0 1 0 1 1 1]
 [0 0 0 1 0 0 0]
 [0 0 0 1 0 0 0]
 [1 0 0 1 0 0 0]]

Node Names: {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
Eccentricity
	of node "A" =   2
	of node "B" =   3
	of node "C" =   2
	of node "D" =   2
	of node "E" =   3
	of node "F" =   3
	of node "G" =   2

Average Distance
	of node "A" = 1.33333
	of node "B" =   2
	of node "C" = 1.5
	of node "D" = 1.16667
	of node "E" =   2
	of node "F" =   2
	of node "G" = 1.66667

Degree Centrality
	of node "A" =   4
	of node "B" =   2
	of node "C" =   3
	of node "D" =   5
	of node "E" =   1
	of node "F" =   1
	of node "G" =   2

K-hop Centrality (k=2)
	of node "A" =   6
	of node "B" =   4
	of node "C" =   6
	of node "D" =   6
	of node "E" =   5
	of node "F" =   5
	of node "G" =   6

K-rank Centrality (k=2)
	of node "A" =   2
	of node "B" =   4
	of node "C" =   3
	of node "D" =   1
	of node "E" =   5
	of node "F" =   5
	of node "G" =   4

Closeness Centrality
	of node "A" = 0.125
	of node "B" = 0.0833333
	of node "C" = 0.111111
	of node "D" = 0.142857
	of node "E" = 0.0833333
	of node "F" = 0.0833333
	of node "G" = 0.1

Betweenness Centrality
	of node "A" =   3
	of node "B" =   0
	of node "C" = 1.5
	of node "D" = 9.5
	of node "E" =   0
	of node "F" =   0
	of node "G" =   0

Stress Centrality
	of node "A" =   5
	of node "B" =   0
	of node "C" =   3
	of node "D" =  12
	of node "E" =   0
	of node "F" =   0
	of node "G" =   0

Clustering Coefficient
	of node "A" = 0.5
	of node "B" =   1
	of node "C" = 0.666667
	of node "D" = 0.2
	of node "E" =   0
	of node "F" =   0
	of node "G" =   1

Examples of Neighbourhood Overlap:
	of nodes "B and E" =   0
	of nodes "B and F" =   0
	of nodes "B and B" =   1
	of nodes "E and C" = 0.333333
	of nodes "D and C" = 0.2
	of nodes "E and E" =   1
	of nodes "D and D" =   1
	of nodes "A and B" = 0.333333
	of nodes "F and E" =   1
	of nodes "B and G" = 0.333333

Examples of Shortest Paths:
	from D to G:  [D -> G (cost: 1)]
	from B to E:  [B -> A -> D -> E (cost: 3), B -> C -> D -> E (cost: 3)]
	from F to F:  [F (cost: 0)]
	from F to B:  [F -> D -> A -> B (cost: 3), F -> D -> C -> B (cost: 3)]
	from G to A:  [G -> A (cost: 1)]
```

## Common dependecies

- python=3.12.2
- numpy=1.26.4
- scipy=1.13.1
- pandas=2.2.2
- matplotlib=3.8.4
- tqdm=4.66.4
- sympy=1.13.3
- regex=2.5.135
- unicodedata=15.0.0
- wordfreq=3.7.0
- spacy=3.7.6
- spellchecker=0.8.0
- bs4=4.12.3

## Contributions

I would appreciate any comments or contributions
