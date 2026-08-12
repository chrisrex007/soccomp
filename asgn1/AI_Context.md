# Assignment 1 Context

**Course**: Social Computing [CS60017] 2026  
**Assignment**: Assignment 1 - Network Analysis  
**Deadline**: August 21, 2026

## General Instructions

1. This assignment utilizes the SNAP High-energy physics theory citation network (cit-HepTh) dataset.
2. **YOUR CODES SHOULD PRINT OUT EXACTLY WHAT IS ASKED, IN THE SPECIFIED FORMAT. DO NOT OUTPUT ANYTHING EXTRA.** The results will be evaluated by an automated checker. There will be a penalty if the specified output format is not followed.
3. **DO NOT USE AI** at any stage to complete this assignment. AI-generated content will be awarded zero for the whole assignment.
4. Write code into different functions with appropriate comments. Ensure code is easily readable. Illegible codes will be penalized.
5. **Submission Format**: Submit one `.zip` or `.tar.gz` file containing:
   - All source codes
   - All required output files
   - `instructions.txt` file (see below)
   - Name the compressed file with your roll number (e.g., `22CS60100.zip`)
6. **instructions.txt File**: Must contain:
   - How to run your code
   - Your name and roll number
   - Python version
   - Any additional information (e.g., if code takes >10 minutes to run, estimate the time)
7. Ensure your code can be run on a computer with 4GB+ RAM by following submitted instructions.
8. **No plagiarism**: Do not copy code from others or web sources.

## Project Goal

This folder contains the scaffold for Social Computing CS60017 Assignment 1.
The task is to analyze the SNAP cit-HepTh citation network after extracting the
subgraph `cite-phy.elist` by removing all nodes whose IDs are divisible by 2.
All graphs must be treated as **undirected**.

## Folder Layout

- `gen structure.py`: structural metrics entry point
- `gen centrality.py`: centrality metrics entry point
- `subgraphs/`: expected location for the extracted subgraph edge list
- `plots/`: expected location for generated PNG plots
- `centralities/`: expected location for centrality output text files
- `.venv/`: project-local Python virtual environment
- `instructions.txt`: submission and run instructions template

## Dataset [5 Marks]

Download the edge list from: https://snap.stanford.edu/data/citHepTh.html

**Subgraph Extraction Rule**: Remove all nodes with IDs divisible by 2

Generate the edge list file with `.elist` extension and save it in the `subgraphs/` folder with filename: `cite-phy.elist`

**NOTE**: Do NOT upload the original dataset file in your submission.

## Command-Line Usage

The scripts are expected to be run as:

```bash
python gen_structure.py cite-phy.elist
python gen_centrality.py cite-phy.elist
```

The scripts should accept the subgraph filename as a single command-line argument.

## TASK 1: Structural Metrics [50 Marks]

**File**: `gen_structure.py`  
**Output**: Print to stdout (no file output except plots)  
**Plots location**: `plots/` folder  
**Rounding**: Round all fractions to **4 decimal places**

### 1. Size of the Network [5 Marks]

- **Number of nodes**: Print line `Number of nodes:<value>`
- **Number of edges**: Print line `Number of edges:<value>`

### 2. Degree of Nodes [5 Marks]

- **Highest degree nodes**: Print line `Node id(s) with highest degree:<comma separated id(s)>`
  - If multiple nodes have the same highest degree, list all of them
- **Degree distribution plot**: Create file `deg_dist_cite-phy.elist.png` in `plots/`

### 3. Paths in the Network [10 Marks]

- **Approximate diameter** by sampling 10, 100, 1000 random test nodes:
  - Print line: `Approximate full diameter by sampling <#test nodes> nodes:<diameter value>`
  - Do this for 10, 100, and 1000 nodes
- **Mean and variance of diameter estimates**:
  - Print line: `Approximate full diameter (mean and variance):<mean value>,<variance value>`
- **Shortest path distribution plot**: Create file `shortest_path_cite-phy.elist.png` in `plots/`

### 4. Components of the Network [16 Marks]

- **Largest connected component fraction**: Print line `Fraction of nodes in largest connected component:<value>`
- **Edge bridges**: Print line `Number of edge bridges:<value>`
  - An edge is a bridge if removing it increases the number of connected components
- **Articulation points**: Print line `Number of articulation points:<value>`
  - A node is an articulation point if removing it increases the number of connected components
- **Connected components distribution plot**: Create file `connected_comp_cite-phy.elist.png` in `plots/`

### 5. Connectivity and Clustering [24 Marks]

- **Average clustering coefficient**: Print line `Average clustering coefficient:<value>`
- **Number of triads**: Print line `Number of triads:<value>`
  - A triad is a set of 3 connected nodes
- **Clustering coefficient of random node**: Print line `Clustering coefficient of random node <node id>:<value>`
  - Include the selected node ID
- **Number of triads for random node**: Print line `Number of triads random node <node id> participates:<value>`
  - Include the selected node ID
- **Edges in triads**: Print line `Number of edges that participate in at least one triad:<value>`
- **Clustering coefficient distribution plot**: Create file `clustering_coeff_cite-phy.elist.png` in `plots/`

## TASK 2: Centrality Metrics [45 Marks]

**File**: `gen_centrality.py`  
**Output location**: `centralities/` folder  
**Output format**: Each file contains lines: `nodeID<whitespace>centrality_value`  
**Rounding**: Round centrality values to **6 decimal places**  
**Sorting**: Nodes in each file must be **sorted by centrality value** (highest to lowest)  
**Max iterations**: Use up to 100 iterations wherever required  
**IMPORTANT**: Do NOT use built-in NetworkX centrality functions (e.g., `betweenness_centrality`). Implement these yourself. Usage of built-in tools will be penalized.

### 1. Closeness Centrality [5 Marks]

**Formula**: $C_i = \frac{1}{\sum_{j=1}^{n-1} d_{ij}}$

Where:

- $d_{ij}$ is the shortest path length from node $i$ to node $j$
- $n$ is the number of nodes

**Output file**: `closeness.txt`

### 2. Betweenness Centrality [10 Marks]

**Formula**: $B_i = \frac{2}{(n-1)(n-2)} \sum_{s,t} \frac{n_i^{st}}{g_{st}}$

Where:

- $n_i^{st}$ is the number of shortest paths between nodes $s$ and $t$ that pass through node $i$
- $g_{st}$ is the total number of shortest paths between nodes $s$ and $t$
- $n$ is the number of nodes

**Output file**: `betweenness.txt`

### 3. Eigenvector Centrality [10 Marks]

**Formula**: $x_i = \frac{1}{\lambda} \sum_{j=1}^{n} A_{ij} x_j$

Where:

- $A_{ij}$ is the adjacency matrix element (connection between nodes $i$ and $j$)
- $x_j$ is the centrality of node $j$
- $\lambda$ is the largest eigenvalue of the adjacency matrix

**Output file**: `eigenvector.txt`

### 4. Biased PageRank [10 Marks]

**Method**: Standard PageRank power-iteration method  
**Damping factor**: $\alpha = 0.8$  
**Preference vector**: Non-uniform, biased towards nodes with IDs divisible by 3  
**Max iterations**: Up to 100

**Output file**: `pagerank.txt`

## Output Requirements

### `gen_structure.py`

- Print only the required stdout lines
- Create plots in `plots/` using exact filenames specified
- No extra text output

### `gen_centrality.py`

- Write four score files into `centralities/` directory:
  - `closeness.txt`
  - `betweenness.txt`
  - `eigenvector.txt`
  - `pagerank.txt`
- Each line format: `nodeID<whitespace>centrality_value`
- Centrality values rounded to 6 decimal places
- Nodes sorted by centrality value (highest first)

## Existing Stub Functions

### `gen_structure.py`

Current stub functions that still need implementation:

- `parse_args()` - Parse command-line argument for subgraph filename
- `ensure_output_directories()` - Create `plots/` directory if it doesn't exist
- `load_subgraph(subgraph_name)` - Load edge list and return networkx graph
- `print_network_size(graph)` - Print number of nodes and edges
- `print_degree_metrics(graph, subgraph_name)` - Print highest degree nodes and create degree distribution plot
- `print_path_metrics(graph, subgraph_name)` - Calculate and print diameter estimates and shortest path plot
- `print_component_metrics(graph, subgraph_name)` - Print component statistics and create component distribution plot
- `print_clustering_metrics(graph, subgraph_name)` - Print clustering statistics and create clustering plot
- `main()` - Orchestrate all metric computations

### `gen_centrality.py`

Current stub functions that still need implementation:

- `parse_args()` - Parse command-line argument for subgraph filename
- `ensure_output_directory()` - Create `centralities/` directory if it doesn't exist
- `load_subgraph(subgraph_name)` - Load edge list and return networkx graph
- `compute_closeness(graph)` - Implement closeness centrality algorithm
- `compute_betweenness(graph)` - Implement betweenness centrality algorithm
- `compute_eigenvector(graph)` - Implement eigenvector centrality algorithm
- `compute_pagerank(graph)` - Implement biased PageRank algorithm
- `write_scores(filename, scores)` - Write centrality scores to file in required format
- `main()` - Orchestrate all centrality computations

## Implementation Constraints

- **Do NOT use built-in NetworkX centrality helpers** (e.g., `betweenness_centrality()`, `closeness_centrality()`, `eigenvector_centrality()`, `pagerank()`). Implement these yourself.
- Keep all graphs **undirected**.
- Keep output formatting **exactly as specified** — an automated checker will compare it.
- Round values **exactly as the assignment specifies**:
  - Structural metrics: 4 decimal places
  - Centrality values: 6 decimal places
- Keep all generated files inside the required folders (`plots/`, `centralities/`, `subgraphs/`).
- **Avoid printing any extra text** beyond what the assignment asks for.
- No AI-generated code.

## Suggested Implementation Order

1. Implement shared graph-loading and edge-list parsing logic (both scripts need this).
2. Implement structural metric helpers in `gen_structure.py`:
   - Start with simple metrics (network size, degree)
   - Then path metrics (diameter, shortest paths)
   - Then component metrics
   - Then clustering metrics
3. Implement centrality routines in `gen_centrality.py` (in order of complexity):
   - Closeness centrality (requires shortest paths)
   - Betweenness centrality (requires all-pairs shortest paths)
   - Eigenvector centrality (requires eigenvalue computation)
   - PageRank (requires power iteration)
4. Add exact output formatting and file writing.
5. Create plots with exact filenames.
6. Re-check all output against the assignment before submission.
