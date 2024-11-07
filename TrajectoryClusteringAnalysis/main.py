import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import  dendrogram,linkage,fcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import numpy as np
import seaborn as sns
from logger import logging
#import logging
from TCA import TCA



def main():
    try:
        df = pd.read_csv('data/mvad_data.csv')
        
        # Map states to numerical values
        state_mapping = {"EM": 2, "FE": 4, "HE": 6, "JL": 8, "SC": 10, "TR": 12}
        colors = ['blue', 'orange', 'green', 'red', 'yellow', 'gray']
        df_numeriques = df.replace(state_mapping)
        
        # Instanciation de l'analyse des trajectoires
        tca = TCA(df_numeriques, state_mapping, colors)
        
        # Effectuer différentes étapes d'analyse et de visualisation
        tca.bar_treatment_percentage()
        tca.plot_treatment_percentages()
        distance_matrix = tca.calculate_distance_matrix()
        linkage_matrix = tca.cluster(distance_matrix)
        
        leaves_order = list(hierarchy.leaves_list(linkage_matrix))
        #***trois lignes suivantes permettent de choisoir le nombre de cluster optimal***#
        #tca.plot_dendrogram(linkage_matrix)
        #tca.plot_clustermap(linkage_matrix)
        #tca.plot_inertia(linkage_matrix)
        
        # Visualisations supplémentaires
        num_clusters = 4
        clusters = tca.assign_clusters(linkage_matrix, num_clusters=num_clusters)
        tca.plot_cluster_heatmaps(clusters, leaves_order, sorted=False)
        tca.plot_treatment_percentages(clusters)
        tca.bar_treatment_percentage(clusters)
        tca.plot_stacked_bar(clusters)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()

