import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import  dendrogram,linkage
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from scipy.spatial.distance import pdist
import numpy as np


##########


class TCA:
    def __init__(self, data):
        self.data = data

    def calculate_distance_matrix(self):
        distance_matrix = pdist(self.data, 'hamming')
        return distance_matrix

    def plot_dendrogram(self, linkage_matrix):
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title('Dendrogram of Treatment Sequences')
        plt.xlabel('Patients')
        plt.ylabel('Distance')
        plt.show()

def main():
    # Exemple d'utilisation
    df = 
    data_array = df_subsets_numeriques.to_numpy()
    tca = TCA(data_array)
    distance_matrix = tca.calculate_distance_matrix()
    linkage_matrix = linkage(distance_matrix, method='ward', optimal_ordering=True)
    tca.plot_dendrogram(linkage_matrix)

if __name__ == "__main__":
    main()