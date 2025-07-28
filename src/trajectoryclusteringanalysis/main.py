import pandas as pd
from tca_dev import TCA


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

    tca = TCA(data=pivoted_data_random_sample,
              index_col='id',
              time_col=None,  # Not used in unidimensional analysis
              event_col=None,  # Not used in unidimensional analysis
              alphabet=['D', 'C', 'T', 'S'],
              states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"], 
              mode='unidimensional',
              )
   
    # Exemple d’utilisation dans une classe TCA ou directement
    labels, centers, inertia = tca.kmeans_on_frequency(
    num_clusters=4,
    random_state=42,
    normalize=False,
)
    print("Labels:", labels)
    print("Centres:", centers)
    print("Inertia:", inertia)
    tca.plot_cluster_heatmaps(tca.data,labels, sorted=True)
    tca.plot_treatment_percentage()
    tca.plot_treatment_percentage(labels)
    tca.bar_treatment_percentage()
    tca.bar_treatment_percentage(labels)
    tca.plot_filtered_heatmap(labels=labels, kernel_size=(0, 0))  # Pas de filtre modal
    tca.plot_filtered_heatmap(labels=labels, kernel_size=(10, 7))  


if __name__ == "__main__":
    main()