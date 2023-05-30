"""
This model analysis file is for the learning path analysis,
it includes building graph model, generating graph embeddings, 
calculting similarities, and some data analysis.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn   
from torch_geometric.nn import GCNConv
import torch_geometric.utils.convert as convert
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Random seed set so that analysis is consistent below
random_seed = 1
torch.manual_seed(random_seed)


def get_all_students(data):
    '''
    Return all students in the data
    '''
    return data['student identifier'].unique()

def calculate_similarity(student_embedding, recommended_embedding):
    '''
    Compute the cosine similarity between the two embeddings
    '''
    similarity = cosine_similarity(student_embedding, recommended_embedding)
    return similarity.mean()


class GCN(nn.Module):
    '''
    Define a GCN model with two layers and a fully connected layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=1) # add node and edge attributes
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        x = torch.sum(x, dim=0) # sum up all the node embeddings to get the graph embedding
        x = self.fc(x)
        return x


class GraphEmbedding:
    '''
    Define a graph embedding method to convert a graph into a embedding
    '''
    def __init__(self, graph):
        self.data = self.convert_to_pyg(graph)
        self.embedding = self.generate_embedding(self.data)

    def convert_to_pyg(self, G):
        '''
        Convert the NetworkX graph to PyTorch Geometric data
        '''
        data = convert.from_networkx(G)

        # Add node features to the data
        data.x = torch.tensor([[i, G.nodes[i]['credits'], G.nodes[i]['college code']] for i in G.nodes()], dtype=torch.float)

        # Add edge features to the data
        edge_attr = np.array([list(G.edges[edge].values()) for edge in G.edges()])
        data.edge_attr = torch.from_numpy(edge_attr).float()
        return data

    def generate_embedding(self, data):
        '''
        Generate a graph embedding with give Pytorch Geometric data
        '''
        num_node_features = data.x.shape[1]
        num_edge_features = data.edge_attr.shape[1]

        model = GCN(2*num_node_features + num_edge_features, num_edge_features, 16)
        emb = model(data.x, data.edge_index, data.edge_attr)
        return emb.detach().numpy()
    
    
class RecommendedPath:
    '''
    Extract recommended orders from the existing dataset for different curriculums, and
    generate the graph embeddings for all the recommened learning path
    '''
    def __init__(self, data, curriculum):
        self.data = data
        self.graph = self.generate_recommended_graph(curriculum)
        
    def generate_recommended_graph(self, curriculum):
        # Get recommended courses and semesters for one curriculm
        curriculum_df = self.data[self.data['curriculum code'] == curriculum]
        curriculum_df = curriculum_df[['college code','curriculum code', 'course', 'recommended semester', 'credits', 'class', 'professor name']]

        # Categorize the data by 'course' and 'recommended semester', others calculates the mean values
        curriculum_df = curriculum_df.groupby(['course', 'recommended semester']).mean().reset_index().sort_values('recommended semester')

        courses = curriculum_df['course'].unique()
        semesters = curriculum_df['recommended semester'].unique()

        # Create an empty directed graph
        G = nx.DiGraph()

        # Add edges to the graph
        i = 0
        semesters_len = len(semesters)
        curr_courses = curriculum_df[curriculum_df['recommended semester'] == semesters[0]]

        while i < semesters_len-1:
            next_courses = curriculum_df[curriculum_df['recommended semester'] == semesters[i+1]]

            curr_len = len(curr_courses['course'])
            next_len = len(next_courses['course'])

            # Add attributes to edges
            for m in range(curr_len):
                curr_df = curr_courses.iloc[m]
                for n in range(next_len):
                    next_df = next_courses.iloc[n]
                    G.add_edge(curr_df['course'], next_df['course'],
                               weight = i+1,
                               curriculum = curr_df['curriculum code'],
                               classes = curr_df['class'],
                               professor = curr_df['professor name'])

            i += 1
            curr_courses = next_courses

        # Add attributes to each course node
        for course in courses:
            course_df = curriculum_df[curriculum_df['course'] == course]
            G.nodes[course]['credits'] = course_df['credits'].values[0]
            G.nodes[course]['college code'] = course_df['college code'].values[0]

        # Visualize a learning path with NetworkX
        # pos = nx.spring_layout(G, seed = 100)
        # nx.draw_networkx(G, pos = pos, with_labels=True)

        return G

class StudentPath:
    '''
    Extract the learning path for each student, and
    generate the graph based on the learning path, and 
    generate the graph embedding from the whole learning graph
    '''
    def __init__(self, data, student):
        self.data = data
        self.student = student
        self.graph = self.generate_student_graph(self.student)
        
    def generate_student_graph(self, student):
        # Add a end node to student records for graph generation
        student = pd.concat([student, pd.DataFrame([[-1] * len(student.columns)], columns=student.columns)], ignore_index=True)
        student.loc[student.index[-1], 'student semester'] = student['student semester'].max() + 1
        
        courses = student['course'].unique()
        semesters = student['student semester'].unique()
        
        # Create an empty directed graph
        G = nx.DiGraph()
        
        # Add edges
        i = 0
        curr_courses = student[student['student semester'] == semesters[0]]

        semesters_len = len(semesters)

        while i < semesters_len-1:
            next_courses = student[student['student semester'] == semesters[i+1]]
            
            curr_len = len(curr_courses['course'])
            next_len = len(next_courses['course'])
            
            # Add attributes to edges
            for m in range(curr_len):
                curr_df = curr_courses.iloc[m]
                for n in range(next_len):
                    next_df = next_courses.iloc[n]
                    G.add_edge(curr_df['course'], next_df['course'],
                               weight = i+1,
                               curriculum = curr_df['curriculum code'],
                               classes = curr_df['class'],
                               enrollment_year = curr_df['enrolment year'],
                               enrollment_semester = curr_df['enrolment semester'],
                               professor = curr_df['professor name'],
                               term = curr_df['term'],
                               final_status = curr_df['final status'],
                               grades = curr_df['grades'],
                               current_status = curr_df['current status'],
                               attempts = curr_df['attempts']
                              )
                               

            i += 1
            curr_courses = next_courses

        # Add attributes to each course node
        for course in courses:
            course_df = student[student['course'] == course]
            G.nodes[course]['credits'] = course_df['credits'].values[0]
            G.nodes[course]['college code'] = course_df['college code'].values[0]
            
        # Visualize the graph G with NetworkX
        # pos = nx.spring_layout(G, seed = 100)
        # nx.draw_networkx(G, pos = pos, with_labels=True)
        
        return G
    
    

# Load preprocessed data
file_path = 'data/all_preprocessed.csv'
processed_data = pd.read_csv(file_path)
print(f'\nLoading data from {file_path}...\n')

# Generate all the recommended learning path embeddings and
# store in a list for repetitive computations
curriculums_list = list(processed_data['curriculum code'].unique())

recommended_embeddings = []
for curriculum_code in curriculums_list:
    recommended_graph = RecommendedPath(processed_data, curriculum_code).graph
    recommended_path = GraphEmbedding(recommended_graph)
    recommended_embeddings.append(recommended_path.embedding)
    

# Generate the actual learning path embeddings for all students and
# calculate their similarities with the recommended embeddings
all_students = get_all_students(processed_data)
similarity_df = processed_data.groupby(['student identifier']).mean().reset_index()
similarity_df = similarity_df[['student identifier', 'grades']]

print('---------------------------------------------------')
print('\nStart calculating graph embeddings ...')
for student_id in all_students:
    selected_student = processed_data[processed_data['student identifier'] == student_id]
    
    student_graph = StudentPath(processed_data, selected_student).graph
    # Skip the student only take in less than (equal to) 2 semesters
    if len(selected_student['student semester'].unique()) <= 2 or len(student_graph.nodes()) > len(student_graph.edges()):
        similarity_df.loc[similarity_df['student identifier'] == student_id, 'similarity'] = 0
        continue
        
    student_path = GraphEmbedding(student_graph)
    
    # The specific curriculum one student is following
    selected_curriculum = selected_student['curriculum code'].unique()[0]
    recommended_em = recommended_embeddings[curriculums_list.index(selected_curriculum)]
    similarity = calculate_similarity([student_path.embedding], [recommended_em])
    
    # Store each similarity for each student
    similarity_df.loc[similarity_df['student identifier'] == student_id, 'similarity'] = similarity
    
print('Similarity calculation finished.\n')


# Correlation analysis
print('---------------------------------------------------')
print('\nStart analysizing...')

similarity_df = similarity_df[similarity_df['similarity'] != 0]

# Visualization correlation between similarity and grades
plt.scatter(similarity_df['similarity'], similarity_df['grades']) 
plt.xlabel('Similarity')
plt.ylabel('Grades')
plt.title('Correlation between grades and similarity')
plt.savefig('fig/all_correlation.png')
plt.show()


# Merge similarities with other original features
student_attrs = processed_data.groupby('student identifier').mean()
new_data = pd.merge(student_attrs, similarity_df, on="student identifier", how="left", suffixes=('', '_y'))
new_data = new_data.drop(['grades_y'], axis=1)
new_data.dropna(inplace=True)

# Hypothesis testing
avg_sim = new_data['similarity'].mean()
print("\nMean of Similarity: {:.3f}".format(avg_sim))

higher = new_data[new_data.loc[:,"similarity"] > avg_sim].loc[:,"grades"]
lower = new_data[new_data.loc[:,"similarity"] < avg_sim].loc[:,"grades"]

t,p = stats.ttest_ind(higher, lower, alternative='greater')
print("One-sided p-value:", p)


# PCA to reduce high dimensions
pca = PCA(n_components=3)
new_reduced = pca.fit_transform(new_data)

# K-Means Clustering
kmc = KMeans(n_clusters=3, n_init=10)
kmc_model = kmc.fit(new_reduced)

# Plot with different coloured clusters and showing cluster centres
colors=["red","blue","green","purple","orange"]

plt.figure(figsize=(10,8))
for i in range(np.max(kmc_model.labels_)+1):
    plt.scatter(new_reduced[kmc_model.labels_==i][:,0], new_reduced[kmc_model.labels_==i][:,1], label=i, c=colors[i], alpha=0.5)
plt.scatter(kmc_model.cluster_centers_[:,0], kmc_model.cluster_centers_[:,1], label='Cluster Centers', c="black", s=200)
plt.title("K-Means Clustering of Student Records",size=20)
plt.xlabel("Principle Component 1", size=16)
plt.ylabel("Principle Component 2", size=16)
plt.legend()
plt.savefig('fig/all_clustering.png')
plt.show()

# Clusters analysis
k = np.max(kmc_model.labels_)+1
df_clusters = [new_data[kmc_model.labels_==i] for i in range(k)]

stat_dict = { 
    'Cluster' : list(range(k)),
    'Size' :    [len(df_clusters[i]) for i in range(k)],
    'Mean grades' :                        [df_clusters[i]['grades'].mean() for i in range(k)],
    'Mean similarity' :                    [df_clusters[i]['similarity'].mean() for i in range(k)],
    'Mean attempts' :                      [df_clusters[i]['attempts'].mean() for i in range(k)],
}
df_cluster_stats = pd.DataFrame(stat_dict)
print('\n', df_cluster_stats)
print('\nAll analysis finished!')
