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

##########



class TCA:
    def __init__(self, data, state_mapping, colors='viridis'):
        self.data = data
        self.state_label = list(state_mapping.keys())
        self.state_numeric = list(state_mapping.values())
        self.colors = colors
        logging.basicConfig(level=logging.INFO)
        
        if len(self.colors) != len(self.state_label):
            logging.error("Number of colors and states mismatch")
            raise ValueError("The number of colors must match the number of states")
        logging.info("TCA object initialized successfully")
        

    def plot_treatment_percentages(self, clusters=None ):
        """
        Plot the percentage of patients under each state over time.
        If clusters are provided, plot the treatment percentages for each cluster.


        Returns:
        None
         
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("self.data should be a pandas DataFrame")
        if clusters is None:
           
            df = self.data.copy()
            # Initialize an empty list to store data for plotting
            plot_data = []

            # Collect data for each treatment
            for treatment, treatment_label,color in zip(self.state_numeric, self.state_label,self.colors):
                treatment_data = df[df.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                plot_data.append(pd.DataFrame({'Month': months, 'Percentage': percentages, 'Treatment': treatment_label}))
                plt.plot(months, percentages, label=f'{treatment_label}', color=color)

            plt.title('Percentage of Patients under Each State Over Time')
            plt.xlabel('Time')
            plt.ylabel('Percentage of Patients')
            plt.legend(title='State')
            plt.show()

        else :
            num_clusters = len(np.unique(clusters))
            colors = self.colors
            events_value = self.state_numeric
            events_keys = self.state_label
            num_rows = (num_clusters + 1) // 2
            num_cols = min(2, num_clusters)

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            if num_clusters == 2:
                axs = np.array([axs])
            if num_clusters % 2 != 0:
                fig.delaxes(axs[-1, -1])

            for cluster_label in range(1, num_clusters + 1):
                cluster_indices = np.where(clusters == cluster_label)[0]
                cluster_data = self.data.iloc[cluster_indices]

                row = (cluster_label - 1) // num_cols
                col = (cluster_label - 1) % num_cols

                ax = axs[row, col]

                for treatment, treatment_label, color in zip(events_value, events_keys, colors):
                    treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                    months = treatment_data.columns
                    percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                    ax.plot(months, percentages, label=f'{treatment_label}', color=color)
                
                ax.set_title(f'Cluster {cluster_label}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Percentage of Patients')
                ax.legend(title='State')
            
            plt.tight_layout()
            plt.show()


    def calculate_distance_matrix(self, metric='hamming'):
        """
        Calculate the distance matrix for the treatment sequences.

        Parameters:
        metric (str): The distance metric to use. Default is 'hamming'.

        Returns:
        distance_matrix (numpy.ndarray): A condensed distance matrix containing the pairwise distances between treatment sequences.
        """
        distance_matrix = pdist(self.data, metric)
        return distance_matrix
    
    def cluster(self, distance_matrix, method='ward', optimal_ordering=True):
        """
        Perform hierarchical clustering on the distance matrix.

        Parameters:
        distance_matrix (numpy.ndarray): A condensed distance matrix containing the pairwise distances between treatment sequences.
        method (str): The linkage algorithm to use. Default is 'ward'.
        optimal_ordering (bool): If True, the linkage matrix will be reordered so that the distance between successive leaves is minimal.

        Returns:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.
        """
        linkage_matrix = linkage(distance_matrix, method=method, optimal_ordering=optimal_ordering)
        return linkage_matrix

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

    def plot_clustermap(self, linkage_matrix):
        """
        Plot a clustermap of the treatment sequences with a custom legend.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.

        Returns:
        None
        """
        plt.figure(figsize=(8, 8))
        sns.clustermap(self.data, cmap=self.colors, metric='hamming', method='ward', row_linkage=linkage_matrix, col_cluster=False, cbar_pos=None)
        
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.colors[i], label=self.state_label[i]) for i in range(len(self.state_label))]
        plt.legend(handles=handles, labels=self.state_label, loc='center', bbox_to_anchor=(0.5, -0.2), ncol=len(self.state_label) // 2)
        
        plt.xlabel("Time")
        plt.ylabel("Patients")
        plt.title("Trajectory of Temporal Vectors")
        plt.show()

    def plot_inertia(self, linkage_matrix):
        """
        Plot the inertia diagram to help determine the optimal number of clusters.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.

        Returns:
        None
        """
        last = linkage_matrix[-10:, 2]
        last_rev = last[::-1]
        idxs = np.arange(2, len(last) + 2)

        plt.figure(figsize=(10, 6))
        plt.step(idxs, last_rev, c="black")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Inertia Diagram")
        plt.show()

    def assign_clusters(self, linkage_matrix, num_clusters):
        """
        Assign patients to clusters based on the dendrogram.

        Parameters:
        linkage_matrix (numpy.ndarray): The linkage matrix containing the hierarchical clustering information.
        num_clusters (int): The number of clusters to form.

        Returns:
        numpy.ndarray: An array of cluster labels assigned to each patient.
        """
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        return clusters
    
    def plot_cluster_heatmaps(self, clusters, leaves_order, sorted=True):
        """
        Plot heatmaps for each cluster, ensuring the data is sorted by leaves_order.

        Parameters:
        clusters (numpy.ndarray): The cluster assignments for each patient.
        leaves_order (list): The order of leaves from the hierarchical clustering.
        sorted (bool): Whether to sort the data within each cluster. Default is True.

        Returns:
        None
        """
        # Reorder the data according to leaves_order
        reordered_data = self.data.iloc[leaves_order]
        reordered_clusters = clusters[leaves_order]

        num_clusters = len(np.unique(clusters))
        cluster_data = {}

        for cluster_label in range(1, num_clusters + 1):
            cluster_indices = np.where(reordered_clusters == cluster_label)[0]
            cluster_df = reordered_data.iloc[cluster_indices]
            if sorted:
                cluster_df = cluster_df.sort_values(by=cluster_df.columns.tolist())
            cluster_data[cluster_label] = cluster_df

        num_rows = (num_clusters + 1) // 2
        num_cols = min(2, num_clusters)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        if num_clusters == 2:
            axs = np.array([axs])

        for i, (cluster_label, cluster_df) in enumerate(cluster_data.items()):
            row = i // num_cols
            col = i % num_cols
            sns.heatmap(cluster_df, cmap=self.colors, cbar=False, ax=axs[row, col])
            axs[row, col].set_title(f'Heatmap du cluster {cluster_label}')
            axs[row, col].set_xlabel('Time')
            axs[row, col].set_ylabel('Patients')

        if num_clusters % 2 != 0:
            fig.delaxes(axs[-1, -1])

        handles = [plt.Rectangle((0, 0), 1, 1, color=self.colors[i], label=self.state_label[i]) for i in range(len(self.state_label))]
        plt.legend(handles=handles, labels=self.state_label, loc='center', bbox_to_anchor=(-0.1, -0.6), ncol=len(self.state_label) // 2)

        plt.tight_layout()
        plt.show()

    

    def bar_treatment_percentage(self, clusters=None):
        """
        Plot the percentage of patients under each state over time using bar plots.
        If clusters are provided, plot the treatment percentages for each cluster.

        Parameters:
        clusters (numpy.ndarray): Cluster assignments for each patient (optional).

        Returns:
        None
        """
        if clusters is None:
            df = self.data.copy()
            # Initialize an empty list to store data for plotting
            plot_data = []

            # Collect data for each treatment
            for treatment, treatment_label,color in zip(self.state_numeric, self.state_label,self.colors):
                treatment_data = df[df.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                plot_data.append(pd.DataFrame({'Month': months, 'Percentage': percentages, 'Treatment': treatment_label}))
                plt.bar(months, percentages, label=f'{treatment_label}', color=color)

            plt.title('Percentage of Patients under Each State Over Time')
            plt.xlabel('Time')
            plt.ylabel('Percentage of Patients')
            plt.legend(title='State')
            plt.show()

        else:
            num_clusters = len(np.unique(clusters))
            num_rows = (num_clusters + 1) // 2  
            num_cols = min(2, num_clusters)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            if num_clusters == 2:
                axs = np.array([axs])
            if num_clusters % 2 != 0:
                fig.delaxes(axs[-1, -1])

            for cluster_label in range(1, num_clusters + 1):
                cluster_indices = np.where(clusters == cluster_label)[0]
                cluster_data = self.data.iloc[cluster_indices]

                row = (cluster_label - 1) // num_cols
                col = (cluster_label - 1) % num_cols

                ax = axs[row, col]

                for treatment, treatment_label, color in zip(self.state_numeric, self.state_label, self.colors):
                    treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                    months = treatment_data.columns
                    percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                    ax.bar(months, percentages, label=f'{treatment_label}', color=color)

                ax.set_title(f'Cluster {cluster_label}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Percentage of Patients')
                ax.legend(title='State')

            plt.tight_layout()
            plt.show()

    
    def plot_stacked_bar(self, clusters):
        """
        Plot stacked bar charts showing the percentage of patients under each treatment over time for each cluster.

        Parameters:
        clusters (numpy.ndarray): The cluster assignments for each patient.

        Returns:
        None
        """
        num_clusters = len(np.unique(clusters))
        num_rows = (num_clusters + 1) // 2  
        num_cols = min(2, num_clusters)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        if num_clusters == 2:
            axs = np.array([axs])
        if num_clusters % 2 != 0:
            fig.delaxes(axs[-1, -1])

        for cluster_label in range(1, num_clusters + 1):
            cluster_indices = np.where(clusters == cluster_label)[0]
            cluster_data = self.data.iloc[cluster_indices]
            
            row = (cluster_label - 1) // num_cols
            col = (cluster_label - 1) % num_cols
            
            ax = axs[row, col]
            
            stacked_data = []
            for treatment in self.state_numeric:
                treatment_data = cluster_data[cluster_data.eq(treatment).any(axis=1)]
                months = treatment_data.columns
                percentages = (treatment_data.apply(lambda x: x.value_counts().get(treatment, 0)) / len(treatment_data)) * 100
                stacked_data.append(percentages.values)
            
            months = range(len(months))
            bottom = np.zeros(len(months))
            for i, data in enumerate(stacked_data):
                ax.bar(months, data, bottom=bottom, label=self.state_label[i], color=self.colors[i])
                bottom += data
            
            ax.set_title(f'Cluster {cluster_label}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Percentage of Patients')
            ax.legend(title='Treatment')
        
        plt.tight_layout()
        plt.show()





def main():
    df = pd.read_csv('data/mvad_data.csv')
    # tranformer vos données en format large si c'est n'est pas le cas 
    state_mapping = {"EM": 2, "FE": 4, "HE": 6, "JL": 8, "SC": 10, "TR": 12}
    colors = ['blue', 'orange', 'green', 'red', 'yellow', 'gray']
    df_numeriques = df.replace(state_mapping)
    tca = TCA(df_numeriques,state_mapping,colors)
   
    tca.bar_treatment_percentage()
    tca.plot_treatment_percentages()
    distance_matrix = tca.calculate_distance_matrix()
    dis = distance_matrix
    linkage_matrix = tca.cluster(dis)  
    leaves_order = list(hierarchy.leaves_list(linkage_matrix))
    #***trois lignes suivantes permettent de choisoir le nombre de cluster optimal***#
    #tca.plot_dendrogram(linkage_matrix)
    #tca.plot_clustermap(linkage_matrix)
    #tca.plot_inertia(linkage_matrix)
    num_clusters=4
    clusters = tca.assign_clusters(linkage_matrix, num_clusters=num_clusters)
    tca.plot_cluster_heatmaps(clusters,leaves_order,sorted=False)
    tca.plot_treatment_percentages(clusters)
    tca.bar_treatment_percentage(clusters)
    tca.plot_stacked_bar(clusters)
if __name__ == "__main__":
    main()