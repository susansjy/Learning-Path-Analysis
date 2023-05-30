# Learning Path Analysis
This is a research project on learning path analysis in student academic records from three degrees (CSI, ADM, ARQ) of the dataset offered by a Brazilian university. 

## Methodology
A graph-based model is proposed to analyze how closely students follow the recommended order provided by the university and to what extent their academic performance are impacted. We applied a graph to represent a student's learning path. Graph embedding technique (Graph Convolution Network) is applied to convert a graph into an low-dimension embedding. Then we calculated cosine similarities between the recommend learning path and the actual one.

## Libraries & Installation
The project is developed in Python 3.10 and anaconda package manager. All the libraries employed including:

* pandas
* numpy
* networkx
* matplotlib.pyplot
* scipy
* torch
* torch.nn  
* torch_geometric.nn
* torch_geometric.utils.convert
* sklearn.metrics.pairwise
* sklearn.decomposition
* sklearn.cluster
* sklearn.preprocessing

Please use `conda install` or `pip install` to install related libraries before running the scripts.

## Data preprocessing
`data_processor.py` implements the data preprocessing, including data cleaning, data transformation and header translation. 

`load_data` load the data from the file path;
`translate_headers` translates the Portuguese headers of the data to English;
`preprocessing(self)` preprocesses the data by filtering, cleaning, normalizing, and validating.

## Model Generation
`model_generation.py` implements is for the learning path analysis, including building graph model, generating graph embeddings, calculting similarities, and some data analysis.

`GCN` defines the GCN model with two layers and a fully connected layer;
`GraphEmbedding` defines a graph embedding method to convert a graph into a embedding;
`RecommendedPath` extracts recommended learning paths for each curriculum;
`StudentPath` extracts the actual learning path for each student.

## Analysis
Correlation analysis, hypothesis testing and K-Means clustering are applied to analyze the relationship between grades and similarities.

## Running 
First run ***`data_processor.py`*** on different degree(s) to get the processed data. All results are stored in the folder `data`.

Then run ***`model_analysis.py`*** on different datasets to get actual learning path graph embeddings and corresponding similarities with the recommended graph embeddings. Analysis results and related figures are stored in folder `fig`.


