import pandas as pd
from tca import TCA


####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################

def main():
    df = pd.read_csv('data/dataframe_test.csv')
    # Sélectionner les colonnes pertinentes pour l'analyse
    selected_cols = df[['id', 'month', 'care_status']]

    # Créer un tableau croisé des données en format large
    #       -> Chaque individu est sur une ligne.
    #       -> Les mesures dans le temps (Temps1, Temps2, Temps3) sont des colonnes distinctes.
    pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    pivoted_data['id'] = pivoted_data.index
    pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]

    # Renommer les colonnes avec un préfixe "month_"
    pivoted_data.columns = ['id'] + ['month_' + str(int(col)+1) for col in pivoted_data.columns[1:]]
    # print(pivoted_data.columns)

    # Sélectionner un échantillon aléatoire de 10% des données
    pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Filter individuals observed for at least 18 months
    valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)

    # Select only the first 18 months for analysis
    valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # print(pivoted_data_random_sample.head())
    # print(data_ready_for_TCA.duplicated().sum())

    # tca = TCA(df_numeriques,state_mapping,colors)
    tca = TCA(data=pivoted_data_random_sample,
              id='id',
              alphabet=['D', 'C', 'T', 'S'],
              states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])
   
    # tca.plot_treatment_percentages(df_numeriques)

    custom_costs = {'D:C': 1, 'D:T': 2, 'D:S': 3, 'C:T': 1, 'C:S': 2, 'T:S': 1}
    #costs = tca.compute_substitution_cost_matrix(method='custom', custom_costs=custom_costs)
    distance_matrix = tca.compute_distance_matrix(metric='hamming', substitution_cost_matrix=None)
    print("distance matrix :\n",distance_matrix)

    linkage_matrix = tca.hierarchical_clustering(distance_matrix)

    # tca.plot_dendrogram(linkage_matrix)
    #tca.plot_clustermap(linkage_matrix)
    # tca.plot_inertia(linkage_matrix)

    clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)
    
    #tca.plot_cluster_heatmaps(clusters, sorted=False)
    # tca.plot_cluster_treatment_percentage(clusters)
    tca.plot_treatment_percentage()
    tca.plot_treatment_percentage(clusters)
    tca.bar_treatment_percentage()
    tca.bar_treatment_percentage(clusters)

if __name__ == "__main__":
    main()