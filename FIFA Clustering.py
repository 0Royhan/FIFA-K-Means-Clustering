import pandas as pd 
import numpy as np
players = pd.read_csv("players_22.csv")

#features to cluster players based on, and cleaning data to drop rows that have null values
features = ["overall", "potential","wage_eur","value_eur","age"]
players = players.dropna(subset=features)
data = players[features].copy() #actual data after cleaning

#scaling data from 1 to 10 so that each column doesnt overly affect clustering
data = ((data - data.min()) / (data.max() - data.min())) *9 + 1

#initialise random centroids, picking a random value from each column in the dataset
def random_centroids(data,k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids,axis=1)    

centroids = random_centroids(data,5)


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x)**2).sum(axis=1))) #calculating distance of each player's features to the features of each centroid
    return distances.idxmin(axis=1) #finding the cluster assignment for each player with regards to the cluster they have the shortest distance from 
   
labels = get_labels(data, centroids) 

#calculating the geometric mean of each cluster to find the new centroids
def new_centroids(data, centroids, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
#plotting each iteration step
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_clusters(data,labels,centroids,iteration):
    pca = PCA(n_components =2)
    data_2d = pca.fit_transform(data)     #transforming the data into 2d data as opposed to 5 dimensional data that we have due to 5 features being assessed
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait = True)
    plt.title(f'Iteration {iteration}')                 #plotting clusters and centroids
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

max_iterations = 100
K = 3
centroids = random_centroids(data,K)
old_centroids = pd.DataFrame()
iteration = 1

while iteration < max_iterations and not centroids.equals(old_centroids):           #if iteration hits the max, stop the loop, and if the centroids havent changed in an updated iteration
    old_centroids = centroids
    labels = get_labels(data, centroids)
    centroids = new_centroids(data,labels, K)
    plot_clusters(data,labels,centroids, iteration)
    iteration += 1
    
print (centroids)
#print( players[labels ==2] [["short_name"]+ features])

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

S_score = silhouette_score(data, labels)
print(f'Silhouette Score: {S_score}')

CH_index = calinski_harabasz_score(data, labels)
print(f'Calinski-Harabasz Index: {CH_index}')

#comparing to scikit learn's algorithm
#from sklearn.cluster import KMeans
#kmeans = KMeans(3)
#kmeans.fit(data)
#centroids = kmeans.cluster_centers_
#pd.DataFrame(centroids, columns=features).T
