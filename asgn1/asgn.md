## Social Computing [CS60017] 2026

Assignment 1: Network Analysis

Deadline: August 21, 2026

## General Instructions

- 1. This assignment utilizes the work (cit-HepTh) dataset. SNAP High-energy physics theory citation net-

- 2. Your codes should print out exactly what is asked, and in the specified format. Do NOT output anything extra that isn’t asked for. The results will be evaluated by an automated checker. There will be a penalty if the specified output format is not followed.

- 4. Try writing codes into different functions and add appropriate comments. Make sure your code can be easily read. Illegible codes will be penalised!

- 6. Along with the source codes and files asked in the tasks, also submit an additional text file called “instructions.txt” where you should state how to run your codes as well as any additional information you want to convey, such as the version of Python. The instructions.txt file should also contain your name and roll number.

- 7. We should be able to run your submitted code in a computer with a reasonable configuration (for instance 4GB or more RAM) by following your submitted in- structions. If any part of your code takes a long time to run (e.g., more than 10 minutes) report that in the instruction file with an estimate of time required.

- 8. The assignment should be done individually by each student. You should not copy any code from one another, or from any web source. Plagiarised codes will be awarded zero for the whole assignment.


## Dataset

Download the edge list of graph from this link as given in Table 1. Generate the sub- graph using the rule given in the same Table, to make them smaller for us to work with. This subgraph will be used throughout this assignment. Generate the edge list file (with “.elist” extensions) of the subgraph and save them in a folder called “subgraphs”. Do NOT upload the original dataset file in your submission. [URL 🔗](https://snap.stanford.edu/data/cit-HepTh.txt.gz)

NOTE: For this assignment, treat all graphs as undirected.

[5 Marks]

| Subgraph | Rule to extract the | Link to download page |
| --- | --- | --- |
| name | subgraph |   |
| cite-phy.elist | Remove all nodes with | https://snap.stanford.edu/data/cit- |
|   | IDs divisible by 2 | HepTh.html |

*Table 1: Rules for extracting subgraph and submission filename*

## Computations

TASK 1: Write a program gen structure.py to generate the structural metrics given below. Your code should take the name of the subgraph (specified in Table 1) as a command line argument. All the output needs to be printed in a standard output, and not to any file. All plots generated should be kept in a folder called “plots”. We will run your code from a terminal with the following command in Python: python gen structure.py cite-phy.elist [URL 🔗](#page-0)

NOTE: For all fractions, round up to 4 decimal places while printing.

- 1. Size of the network

- (a) Number of nodes

Your code should print the following line to stdout:

Number of nodes:⟨value⟩

- (b) Number of edges

Your code should print the following line to stdout:

Number of edges:⟨value⟩

- 2. Degree of nodes in the network

- (a) Node id(s) for the node with the highest degree. Note that there might be multiple nodes with highest degree.

Your code should print the following line to stdout:

Node id(s) with highest degree:⟨comma separated id(s) of nodes⟩

- (b) Plot of the degree distribution

Your code should create the plotted image in the “plots” directory in a file named: “deg dist ⟨subgraph name⟩.png”

- 3. Paths in the network

[95 Marks]

[5 Marks]

[5 Marks]

[10 Marks]


- (a) Approximate full diameter (maximum shortest path length) computed start- ing from 10, 100, 1000 random test nodes. Also calculate the average and variance across these 3 estimates of the diameter.

Your code should print lines in stdout: Approximate full diameter by sampling ⟨#test nodes⟩ nodes:⟨diameter value⟩ and in the next line: Approximate full diameter (mean and variance):⟨mean value⟩, ⟨variance

value⟩

- (b) Plot of the distribution of the shortest path lengths in the network. Your code should create the plotted image in the “plots” directory in a file named: shortest path ⟨subgraph name⟩.png

- 4. Components of the network

- [16 Marks]

- (a) Fraction of nodes in the largest connected component.

Your code should print the following line to stdout:

Fraction of nodes in largest connected component:⟨value⟩

- (b) Number of edge bridges: An edge is a bridge if, when removed, increases the number of connected components.

## Your code should print the following line to stdout: Number of edge bridges:⟨value⟩

- (c) Number of articulation points: A node is a articulation point if, when removed, increases the number of connected components.

Your code should print the following line to stdout: Number of articulation points:⟨value⟩

- (d) Plot of the distribution of sizes of connected components.

Your code should create the plotted image in the “plots” directory in a file named: connected comp ⟨subgraph name⟩.png

- 5. Connectivity and clustering in the network [24 Marks]

- (a) Average clustering coefficient of the network.

Your code should print the following line to stdout:

Average clustering coefficient:⟨value⟩

- (b) Number of triads

Your code should print the following line to stdout:

Number of triads:⟨value⟩

- (c) Clustering coefficient of a randomly selected node. Also report the selected node id.

## Your code should print the following line to stdout:

Clustering coefficient of random node ⟨node id⟩:⟨value⟩

- (d) Number of triads a randomly selected node participates in. Also report the selected node id.

## Your code should print the following line to stdout:

Number of triads random node ⟨node id⟩ participates:⟨value⟩

- (e) Number of edges that participate in at least one triad

Your code should print the following line to stdout:

Number of edges that participate in at least one triad:⟨value⟩


- (f) Plot of the distribution of clustering coefficient

## Your code should create the plotted image in the “plots” directory in a file named: clustering coeff ⟨subgraph name⟩.png

## TASK 2: Your code file should be named gen centrality.py and keeping the output folders and execution format same as TASK 1 to compute the following centrality metrics

for the obtained graph (use max iterations of upto 100 wherever required):

- 1. Closeness centrality for node i, given by Ci = Pn−1 j dij , where dij is the length of the shortest path from i to j, and n is the number of nodes in the graph. [5Marks]

- 2. Betweenness centrality for node i, given by Bi = (n−1)(n−2) the number of shortest paths between nodes s and t which pass through i, and gst is the total number of shortest paths between nodes s and t. 2 P s,t ni gst st where ni st is [10 Marks]

- 3. Eigenvector centrality for node i, given by xi = element of the adjacency matrix representing the connection between nodes i and j, xj is the centrality of node j, and λ is the largest eigenvalue of the adjacency matrix. λ 1 Pn j=1 Aijxj, where Aij is the [10 Marks]

- 4. Biased PageRank for a node i calculated using the standard PageRank power- iteration method (as discussed in class). Use a non-uniform preference vector biased towards nodes that have their node IDs divisible by 3. Use damping factor, α = 0.8. [10 Marks]

You should output a text file each for the centrality measures, which contains a line for each of the node. Name the output files “closeness.txt”, “betweenness.txt”, “eigen- vector.txt” and “pagerank.txt”. Generate the files inside a folder called “centrali- ties”. Each line in the output files has the format: nodeID⟨whitespace⟩centrality value. The centrality values can be rounded up to 6 decimal places. The nodes in each file should be sorted by centrality value. Note that there are already built-in functions in networkx to calculate all these centrali- ties (example: “betweenness centrality”). However, you should NOT use these functions. Usage of such built-in tools will be penalised! You need to implement these

centralities yourself, as extensions to networkx’s graph framework.

For any queries regarding the assignment, contact TA Subham Kumar (email id on course website).
