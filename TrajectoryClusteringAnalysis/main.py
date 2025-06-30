import pandas as pd
from tca import TCA

####################################### MAIN #######################################

# Fonction principale pour exécuter l'analyse de clustering de trajectoires
def main():
    # Chargement des données depuis un fichier CSV
    df = pd.read_csv('data/dataframe_test.csv')

    # Sélectionner les colonnes pertinentes pour l'analyse
    selected_cols = df[['id', 'month', 'care_status']]

    # Transformation des données en format large pour l'analyse
    # Chaque individu est sur une ligne, et les mesures dans le temps sont des colonnes distinctes
    pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    pivoted_data['id'] = pivoted_data.index
    pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]

    # Renommer les colonnes avec un préfixe "month_"
    pivoted_data.columns = ['id'] + ['month_' + str(int(col) + 1) for col in pivoted_data.columns[1:]]

    # Sélectionner un échantillon aléatoire de 10% des données
    pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Filtrer les individus observés pendant au moins 18 mois
    valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)

    # Sélectionner uniquement les 18 premiers mois pour l'analyse
    valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # Initialisation de l'objet TCA avec les données préparées
    tca = TCA(
        data=pivoted_data_random_sample,
        id='id',
        alphabet=['D', 'C', 'T', 'S'],  # États possibles
        states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"]
    )
   
    # Calcul de la matrice de distances avec la métrique Hamming
    distance_matrix = tca.compute_distance_matrix(metric='dtw', substitution_cost_matrix=None)
    print("distance matrix :\n", distance_matrix)
    kmedoids_labels, medoid_indices, inertia = tca.kmedoids_clustering(distance_matrix, num_clusters=4,method='fasterpam')
    print("kmedoids_labels :\n", kmedoids_labels)
    print("medoid_indices :\n", medoid_indices)
    print("inertia :\n", inertia)
    # Clustering hiérarchique
    #linkage_matrix = tca.hierarchical_clustering(distance_matrix)

    # Attribution des clusters
   # clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)

    # Visualisation des résultats
    tca.plot_cluster_heatmaps(kmedoids_labels, sorted=True)
    tca.plot_treatment_percentage()
    tca.plot_treatment_percentage(kmedoids_labels)
    tca.bar_treatment_percentage()
    tca.bar_treatment_percentage(kmedoids_labels)
    tca.plot_filtered_heatmap(labels=kmedoids_labels, kernel_size=(0, 0))  # Pas de filtre modal
    tca.plot_filtered_heatmap(labels=kmedoids_labels, kernel_size=(10, 7))  

if __name__ == "__main__":
    # Exécution du script principal
    main()