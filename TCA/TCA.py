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
    
    def plot_treatment_percentages(df, state, colors='viridis'):

        """
        Plot the percentage of patients under each state over time.

        Parameters:
        df (pd.DataFrame): DataFrame (format STS) containing the treatment data.
        state (list): List of state.
        colors (list): List of colors corresponding to each state.

        Returns:
        None
        """
        # Create a figure
        fig = go.Figure()

        # Iterate over treatments
        for treatment, color in zip(state, colors):
            # Extract data for the current treatment
            treatment_data = df[df.eq(treatment).any(axis=1)]
            # Get the columns representing months
            months = treatment_data.columns
            # Calculate the percentage of patients under the current treatment for each month
            percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
            # Plot the curve for the percentage of patients under the treatment over time
            fig.add_trace(go.Scatter(x=months, y=percentages, mode='lines', name=treatment, line=dict(color=color)))

        # Update layout
        fig.update_layout(
            title='Percentage of Patients under Each State Over Time',
            xaxis_title='Time',
            yaxis_title='Percentage of Patients',
            legend_title='State',
            yaxis=dict(tickformat=".2f")
        )

        # Display the graph
        fig.show()

    def calculate_distance_matrix(self, metric='hamming'):
        """
        Calculate the distance matrix for the treatment sequences.

        Returns:
        distance_matrix (numpy.ndarray): A condensed distance matrix containing the pairwise distances between treatment sequences.
        """
        data_array = self.data.to_numpy()
        distance_matrix = pdist(self.data, metric)
        return distance_matrix

    def plot_dendrogram(self, linkage_matrix):
        """
        Plot a dendrogram based on the hierarchical clustering of treatment sequences.
        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.
        Returns:
        None
        """
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix)
        plt.title('Dendrogram of Treatment Sequences')
        plt.xlabel('Patients')
        plt.ylabel('Distance')
        plt.show()

def main():
    df = pd.read_csv('data/mvad_data.csv')
    # tranformer vos données en format large si c'est n'est pas le cas 
    state_mapping = {"EM": 2, "FE": 4, "HE": 6, "JL": 8, "SC": 10, "TR": 12}
    df_numeriques = df.replace(state_mapping)
    state_label = list(state_mapping.keys())
    state_numerique = list(state_mapping.values())
    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'gray']
    TCA.plot_treatment_percentages(df, state_label, colors)
    tca = TCA(df_numeriques)
    distance_matrix = tca.calculate_distance_matrix()
    linkage_matrix = linkage(distance_matrix, method='ward', optimal_ordering=True)
    tca.plot_dendrogram(linkage_matrix)

if __name__ == "__main__":
    main()