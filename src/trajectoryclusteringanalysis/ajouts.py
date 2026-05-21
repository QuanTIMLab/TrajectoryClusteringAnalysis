import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from lifelines import KaplanMeierFitter

# Passage de format long à large
def long_to_wide_format(df, id_col, time_col, value_col, prefix="time_", static_cols=None, shift_time=False, sample_frac=0.1, random_state=42):
    """
    Transforme un DataFrame du format long au format large.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame d'origine au format long.
    id_col : str
        Le nom de la colonne identifiant les patients/individus (ex: 'id_patient').
    time_col : str
        Le nom de la colonne temporelle (ex: 'month', 'time').
    value_col : str
        Le nom de la colonne contenant les états/valeurs (ex: 'care_status', 'etat').
    prefix : str, optionnel
        Le préfixe à ajouter aux nouvelles colonnes temporelles créées (défaut: "time_").
    static_cols : list of str, optionnel
        Les ariables cliniques à extraire
        Les noms des colonnes statiques à conserver dans le DataFrame pivoté.
    shift_time : bool, optionnel
        Si True, ajoute +1 à la valeur de la colonne temporelle lors du renommage 
        (utile si les temps commencent à 0 dans les données mais qu'on souhaite afficher M1, M2...).
    sample_frac : float, optionnel
        Fraction de l'échantillon d'individus à conserver (entre 0.0 et 1.0). 
        Défaut: 0.1 (conserve 10%).
    random_state : int, optionnel
        Graine aléatoire (défaut: 42).
        
    Retourne:
    ---------
    pd.DataFrame
        Le DataFrame pivoté au format wide, avec l'identifiant en première colonne
        suivi des colonnes temporelles nommées avec le préfixe
    """
    
    # Tirage au sort des identifiants
    # On isole les IDs uniques
    unique_ids = df[id_col].unique()
    
    # Calcul du nombre de patients à garder
    n_samples = max(1, int(len(unique_ids) * sample_frac)) 
    
    rng = np.random.default_rng(random_state)
    sampled_ids = rng.choice(unique_ids, size=n_samples, replace=False)
    
    # On filtre le DataFrame long pour ne garder que les patients tirés au sort
    df_sampled = df[df[id_col].isin(sampled_ids)]
    
    # On garde les colonnes statiques
    df_clinical = None
    if static_cols is not None:
        cols_to_keep = [id_col] + static_cols
        df_clinical = df_sampled[cols_to_keep].drop_duplicates(subset=[id_col])
        
        df_clinical = df_clinical.drop(columns=[id_col]).reset_index(drop=True)
    
    # Pivot
    pivoted_data = df_sampled.pivot(index=id_col, columns=time_col, values=value_col)
    
    if shift_time:
        pivoted_data.columns = [f"{prefix}{int(col) + 1}" for col in pivoted_data.columns]
    else:
        pivoted_data.columns = [f"{prefix}{col}" for col in pivoted_data.columns]
        
    # Reset index à la toute fin pour repasser l'id_col en colonne normale
    if static_cols is not None:
        return pivoted_data.reset_index(), df_clinical
    else:
        return pivoted_data.reset_index()
    

# Visualisation de la matrice de coûts
def plot_cost_matrix(cost_matrix, ax=None, title="Matrice de coûts de substitution", cmap="YlOrRd", annot_fmt=".2f"):
    """
    Affiche la matrice de coûts sous forme de carte de chaleur (Heatmap).
    
    Paramètres:
    -----------
    cost_matrix : pd.DataFrame
        La matrice des coûts de substitution.
    ax : matplotlib.axes.Axes, optionnel
        L'axe sur lequel tracer le graphique. Si None, une nouvelle figure est créée.
    title : str, optionnel
        Le titre du graphique.
    cmap : str ou matplotlib.colors.Colormap, optionnel
        La palette de couleurs utilisée pour la heatmap (défaut : "YlOrRd").
    annot_fmt : str, optionnel
        Le formatage des nombres affichés dans les cases (défaut : ".2f" pour 2 décimales).
        
    Retourne:
    ---------
    matplotlib.axes.Axes
        L'axe contenant le graphique.
    """
    
    # Si on n'a pas fourni de case spécifique (ax), on en crée une.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        is_autonomous = True
    else:
        is_autonomous = False
        
    # Création de la Heatmap
    sns.heatmap(
        cost_matrix, 
        annot=True, 
        cmap=cmap, 
        cbar_kws={'label': 'Coût de substitution'},
        fmt=annot_fmt,
        linewidths=0.5, 
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel("État d'arrivée", fontsize=12)
    ax.set_ylabel("État de départ", fontsize=12)
    
    # Rotation des étiquettes (ticks) pour la lisibilité
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    
    if is_autonomous:
        plt.tight_layout()
        plt.show()
        
    return ax


def plot_transversal_entropy(tca, clusters=None, title="Évolution de l'Entropie Transversale"):
        """
        Calcule et affiche l'entropie transversale (seqHtplot de TraMineR).
        Si 'clusters' est fourni, affiche une courbe par cluster.

        Paramètres:
        -----------
        tca : TCA object
            L'objet TCA contenant les données de séquences.
        clusters : array-like, optionnel
            Les labels de cluster pour chaque patient. Si None, affiche la moyenne pour l'ensemble des données.
        """        
        df_seqs = tca.data.drop(columns=[tca.index_col])
        time_points = df_seqs.columns
        n_states = len(tca.alphabet)

        def calculate_entropy(df):
            entropies = []
            for col in df.columns:
                # Calcul des proportions p_i pour chaque état
                probs = df[col].value_counts(normalize=True)
                # Formule de Shannon : -sum(p_i * log(p_i))
                # On normalise par log(n_states) pour avoir un score entre 0 et 1
                h = -np.sum(probs * np.log(probs)) / np.log(n_states) if n_states > 1 else 0
                entropies.append(h)
            return entropies

        plt.figure(figsize=(12, 6))

        if clusters is None:
            # Entropie globale
            overall_h = calculate_entropy(df_seqs)
            plt.plot(time_points, overall_h, label="Cohorte Totale", color='black', linewidth=3)
        else:
            # Une courbe par cluster
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                cluster_data = df_seqs[clusters == cluster]
                cluster_h = calculate_entropy(cluster_data)
                plt.plot(time_points, cluster_h, label=f"Cluster {cluster}")

        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Temps (Mois)")
        plt.ylabel("Indice d'Entropie (Normalisé)")
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Groupes", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_mean_time(tca, clusters=None, title="Temps Moyen passé dans chaque État"):
        """
        Calcule et affiche le temps moyen passé dans chaque état (seqmtplot).
        """
        df_seqs = tca.data.drop(columns=[tca.index_col])
        
        def get_means(df):
            # Si le filtre a renvoyé un tableau vide à cause des index, on gère directement
            if df.empty:
                return pd.Series(0, index=tca.alphabet)
                
            counts = df.apply(pd.Series.value_counts, dropna=False, axis=1).fillna(0)
            for state in tca.alphabet:
                if state not in counts.columns:
                    counts[state] = 0
            
            # AJOUT SECURITÉ : .fillna(0) au cas où la moyenne génère un NaN
            return counts[tca.alphabet].mean().fillna(0)

        plt.figure(figsize=(10, 6))
        
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(tca.alphabet))]

        if clusters is None:
            means = get_means(df_seqs)
            bars = plt.bar(tca.states, means, color=colors, edgecolor='black', alpha=0.8)
            plt.ylabel("Nombre moyen de mois")
        else:
            # Comparaison par cluster (Barres groupées)
            # AJOUT SECURITÉ : on s'assure que unique_clusters lit bien les valeurs nettes
            unique_clusters = np.unique(np.array(clusters))
            n_clusters = len(unique_clusters)
            width = 0.8 / n_clusters
            x = np.arange(len(tca.states))

            for i, cluster in enumerate(unique_clusters):
                # AJOUT SECURITÉ : le ".values" force Pandas à ignorer le conflit d'index des patients
                cluster_means = get_means(df_seqs[np.array(clusters) == cluster])
                plt.bar(x + i*width, cluster_means, width, label=f"Cluster {cluster}", alpha=0.8)
            
            plt.xticks(x + width*(n_clusters-1)/2, tca.states)
            plt.ylabel("Durée moyenne (Mois)")
            plt.legend(title="Groupes", bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def extract_representative_sequences(df_sequences, distance_matrix, cluster_labels, coverage_threshold):
    """
    Identifie la séquence médoïde de chaque cluster et calcule sa représentativité.
    
    Paramètres:
    -----------
    df_sequences : DataFrame
        Le dataframe contenant uniquement les séquences (sans la colonne d'index patient).
    distance_matrix : ndarray
        La matrice de distance.
    cluster_labels : array-like
        Les labels de cluster affectés à chaque patient (taille N).
    coverage_threshold : float
        La distance maximale acceptable pour dire qu'un patient est "représenté" par le médoïde.
        
    Retourne:
    ---------
    dict : Un dictionnaire contenant les infos du médoïde pour chaque cluster.
    """
    unique_clusters = np.unique(cluster_labels)
    results = {}
    
    for cluster in unique_clusters:
        idx_cluster = np.where(cluster_labels == cluster)[0]
        n_patients = len(idx_cluster)
        
        # Extraire la sous-matrice de distance uniquement pour ce cluster
        sub_matrix = distance_matrix[np.ix_(idx_cluster, idx_cluster)]
        
        # Trouver le médoïde (celui dont la somme des distances aux autres est minimale)
        sum_distances = sub_matrix.sum(axis=1)
        medoid_relative_idx = np.argmin(sum_distances) # Index dans la sous-matrice
        medoid_absolute_idx = idx_cluster[medoid_relative_idx] # Index dans le jeu de données global
        
        # Calculer le taux de couverture
        distances_to_medoid = sub_matrix[medoid_relative_idx]
        covered_count = np.sum(distances_to_medoid <= coverage_threshold)
        coverage_rate = (covered_count / n_patients) * 100
        
        # Stocker les résultats
        results[cluster] = {
            'medoid_index': medoid_absolute_idx,
            'sequence': df_sequences.iloc[medoid_absolute_idx].values,
            'n_patients': n_patients,
            'coverage_rate': coverage_rate,
            'covered_count': covered_count
        }
        
    return results


def plot_representative_sequences(tca, rep_seq_results, title="Séquences Représentatives (Médoïdes)"):
    """
    Affiche les séquences représentatives avec leur taux de couverture.
    Paramètres:
    -----------
    tca : TCA object
    rep_seq_results : dict
        Dictionnaire contenant les informations des séquences représentatives pour chaque cluster.
    title : str
        Le titre du graphique. 
    """
    clusters = list(rep_seq_results.keys())
    n_clusters = len(clusters)
    
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 1.2 * n_clusters), sharex=True)
    if n_clusters == 1: axes = [axes]
        
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(tca.alphabet))]
    custom_cmap = mcolors.ListedColormap(colors)
    custom_cmap.set_bad(color='lightgray')
    bounds = np.arange(len(tca.alphabet) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    
    state_to_idx = {state: idx for idx, state in enumerate(tca.alphabet)}
    
    for idx, cluster in enumerate(clusters):
        ax = axes[idx]
        info = rep_seq_results[cluster]
        
        # Convertir la séquence en numérique pour l'affichage
        seq_numeric = np.array([state_to_idx.get(s, np.nan) for s in info['sequence']])
        matrix = seq_numeric.reshape(1, -1).astype(float) # Format requis pour imshow
        
        # Affichage
        ax.imshow(matrix, aspect='auto', cmap=custom_cmap, norm=norm, interpolation='none')
        
        # Labels axe Y : Nom du cluster et couverture
        label = (f"Cluster {cluster}\n"
                 f"Couverture: {info['coverage_rate']:.1f}%\n"
                 f"({info['covered_count']}/{info['n_patients']} patients)")
        
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=10)
        ax.tick_params(axis='y', length=0)
        
    df_time_base = tca.data.drop(columns=[tca.index_col])
    x_labels = df_time_base.columns
    step = 3
    axes[-1].set_xticks(np.arange(0, len(x_labels), step))
    axes[-1].set_xticklabels(x_labels[::step], rotation=45, ha='right')
    axes[-1].set_xlabel("Temps")

    patches = [mpatches.Patch(color=colors[i], label=tca.states[i]) for i in range(len(tca.alphabet))]
    patches.append(mpatches.Patch(color='lightgray', label="Pas de donnée"))
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(len(tca.states), 5), frameon=False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def extract_coverage_kmedoids(df_sequences, distance_matrix, cluster_labels, medoids_indices, coverage_threshold):
    """
    Calcule le taux de couverture des médoïdes (donné par k-medoids).
    
    Paramètres:
    -----------
    df_sequences : DataFrame
        Le dataframe contenant uniquement les séquences.
    distance_matrix : ndarray
        La matrice de distance globale.
    cluster_labels : array-like
        Les labels de cluster affectés à chaque patient.
    medoids_indices : array-like
        Les index absolus des médoïdes (généralement kmedoids_model.medoid_indices_).
    coverage_threshold : float
        La distance maximale acceptable pour le voisinage.
        
    Retourne:
    ---------
    dict : Un dictionnaire contenant les infos du médoïde pour chaque cluster.
    """
    unique_clusters = np.unique(cluster_labels)
    results = {}
    
    # Création d'un dictionnaire pour associer chaque médoïde à son cluster
    medoid_dict = {cluster_labels[idx]: idx for idx in medoids_indices}
    
    for cluster in unique_clusters:
        # Trouver tous les patients appartenant à ce cluster
        idx_cluster = np.where(cluster_labels == cluster)[0]
        n_patients = len(idx_cluster)
        
        # Récupérer l'index du médoïde de ce cluster
        medoid_absolute_idx = medoid_dict[cluster]
        
        # Extraire directement les distances entre ce médoïde et les patients de son cluster
        distances_to_medoid = distance_matrix[medoid_absolute_idx, idx_cluster]
        
        # Calculer le taux de couverture
        covered_count = np.sum(distances_to_medoid <= coverage_threshold)
        coverage_rate = (covered_count / n_patients) * 100
        
        # Stocker les résultats
        results[cluster] = {
            'medoid_index': medoid_absolute_idx,
            'sequence': df_sequences.iloc[medoid_absolute_idx].values,
            'n_patients': n_patients,
            'coverage_rate': coverage_rate,
            'covered_count': covered_count
        }
        
    return results


def plot_trajectory_decision_tree(df_clinical, cluster_labels, max_depth=3, variables_label=None, titre="Arbre de décision des trajectoires cliniques"):
    """
    Entraîne et affiche un arbre de décision épuré pour voir quelles variables
    conduisent aux différents clusters (avec de vrais labels explicites).
    """
    X = df_clinical.copy()
    
    # 1. On traduit les chiffres en vrais mots (Homme, Femme, Pauvre...)
    if variables_label is not None:
        X = X.replace(variables_label)
        
    # 2. On crée les colonnes binaires (ex: sex_Femme) pour l'arbre
    X = pd.get_dummies(X)
        
    y = cluster_labels
    
    # Entraînement de l'arbre
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=0.05)
    clf.fit(X, y)
    
    # Visualisation
    plt.figure(figsize=(15, 8))
    
    feature_noms = X.columns.tolist()
    class_names = [f"Cluster {c}" for c in clf.classes_]
    
    annotations = plot_tree(
        clf, 
        feature_names=feature_noms,
        class_names=class_names,
        filled=True, 
        rounded=True, 
        fontsize=12,
        impurity=False,
        proportion=True
    )
    
    
    for text_obj in annotations:
        text = text_obj.get_text()
        lignes = text.split('\n')
        lignes_new = []
        
        for ligne in lignes:
            if "samples" in ligne or "value" in ligne:
                continue
                
            if "<=" in ligne:
                col_name = ligne.split(" <= ")[0]
                
                if "_" in col_name and "0.5" in ligne:
                    var_name, modalite = col_name.split("_", 1)
                    lignes_new.append(f"{var_name} : {modalite} ?")
                    # Ajout de la boussole anti-erreur
                    lignes_new.append("<- Non  | Oui ->")
                else:
                    lignes_new.append(f"{ligne} ?")
            else:
                lignes_new.append(ligne)
                
        nouveau_texte = '\n'.join(lignes_new)
        text_obj.set_text(nouveau_texte)
        text_obj.set_fontweight('bold')

    plt.title(titre, fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return clf


def plot_time_to_event(df_sequences, cluster_labels, target_state='S', time_unit="Mois", titre="Temps avant d'atteindre l'état S par cluster"):
    """
    Affiche les courbes de Kaplan-Meier (Incidence Cumulée) pour comparer 
    la rapidité avec laquelle chaque cluster atteint l'état S.

    Paramètres :
    -----------
    df_sequences : pd.DataFrame
        DataFrame contenant les séquences d'états (sans la colonne d'identifiant).
    cluster_labels : array-like
        Les labels de cluster pour chaque patient.
    target_state : str, optionnel
        L'état d'intérêt pour l'analyse de survie (défaut : 'S').
    time_unit : str, optionnel
        L'unité de temps à afficher sur l'axe X (défaut : "Mois").
    """
    T = [] # Temps jusqu'à l'événement (ou temps de suivi max si non atteint)
    E = [] # 1 si l'état S est atteint, 0 sinon
    
    max_time = df_sequences.shape[1]
    
    # Extraction des temps d'atteinte pour chaque patient
    for i in range(len(df_sequences)):
        seq = df_sequences.iloc[i].values
        # Première occurrence de l'état S
        indices = np.where(seq == target_state)[0]
        
        if len(indices) > 0:
            T.append(indices[0]) # Le temps est l'index de la première occurrence
            E.append(1)          # L'événement s'est produit
        else:
            T.append(max_time)   # N'a jamais atteint l'état pendant le suivi
            E.append(0)          # L'événement ne s'est pas produit
            
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        # Filtrer les patients appartenant à ce cluster
        idx = (cluster_labels == cluster)
        T_cluster = np.array(T)[idx]
        E_cluster = np.array(E)[idx]
        
        kmf.fit(T_cluster, event_observed=E_cluster, label=f"Cluster {cluster}")
        
        # plot_cumulative_density affiche la proportion qui a ATTEINT l'état (monte de 0 à 1)
        kmf.plot_cumulative_density(ci_show=False, linewidth=2.5)

    # 4. Esthétique du graphique
    plt.title(titre, fontsize=15, pad=15, fontweight='bold')
    plt.xlabel(f"Temps écoulé ({time_unit})", fontsize=12)
    plt.ylabel(f"Proportion de patients dans l'état '{target_state}'", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # On force l'axe Y à aller de 0% à 100%
    plt.ylim(0, 1.05)
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([f"{int(x*100)}%" for x in yticks])
    
    plt.legend(
        title="Clusters", 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), 
        ncol=min(len(unique_clusters), 5), 
        frameon=True
        )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_cluster_profiles(df_clinical, cluster_labels, variables_label=None, variables_to_plot=None, titre="Profil clinique des clusters"):
    """
    Affiche le profil clinique complet sous forme de barres empilées horizontales.

    Paramètres:
    -----------
    df_clinical : pd.DataFrame  
        DataFrame contenant les variables cliniques (sans la colonne d'identifiant).
    cluster_labels : array-like
        Les labels de cluster pour chaque patient.
    variables_label : dict, optionnel
        Un dictionnaire pour remplacer les valeurs numériques par leur vrai nom.
    variables_to_plot : list, optionnel
        Une liste de colonnes spécifiques à inclure dans le graphique. Si None, toutes les colonnes seront utilisées.
    titre : str, optionnel
        Le titre du graphique.
    """
    df_analyse = df_clinical.copy()

    if variables_label is not None:
        df_analyse = df_analyse.replace(variables_label)

    df_analyse['Cluster'] = cluster_labels
    
    if variables_to_plot is None:
        variables_to_plot = [col for col in df_analyse.columns if col != 'Cluster']
        
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    all_modalities = []
    for var in variables_to_plot:
        for mod in sorted(df_analyse[var].dropna().unique()):
            if mod not in all_modalities:
                all_modalities.append(mod)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(11, 3.5 * n_clusters), sharex=True)
    if n_clusters == 1: 
        axes = [axes]
        
    for i, cluster in enumerate(unique_clusters):
        ax = axes[i]
        df_c = df_analyse[df_analyse['Cluster'] == cluster]
        
        cluster_profiles = []
        for var in variables_to_plot:
            counts = df_c[var].value_counts(normalize=True) * 100
            counts.name = var
            cluster_profiles.append(counts)
        
        df_plot = pd.concat(cluster_profiles, axis=1).T
        df_plot = df_plot.reindex(columns=all_modalities).fillna(0)
        
        df_plot.plot(
            kind='barh', 
            stacked=True, 
            ax=ax, 
            colormap='tab20', 
            edgecolor='white', 
            alpha=0.85,
            legend=False 
        )
        
        for patch in ax.patches:
            width = patch.get_width()
            height = patch.get_height()
            x = patch.get_x()
            y = patch.get_y()
            
            # Si le segment représente plus de 5% de la barre, on met le texte
            if width > 3:
                col_idx = int(np.floor(ax.patches.index(patch) / len(variables_to_plot)))
                if col_idx < len(df_plot.columns):
                    label_text = df_plot.columns[col_idx]
                    
                    # On place le texte au milieu du segment
                    ax.text(
                        x + width/2, 
                        y + height/2, 
                        f"{label_text}\n({width:.1f}%)", 
                        ha='center', 
                        va='center', 
                        color='black',      
                        fontweight='bold',   # En gras
                        fontsize=9         
                    )
        # -----------------------------------------------------------
        
        ax.set_title(f"Profil des patients : Cluster {cluster} (n = {len(df_c)} patients)", fontsize=13, pad=10)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    axes[-1].set_xlabel("Proportion au sein du cluster (%)", fontsize=11)
    plt.tight_layout()
    plt.suptitle(titre, fontsize=14, fontweight='bold', y=1.01)
    plt.show()


def plot_frequent_transitions(tca, clusters, n_top=6, titre="Transitions les plus fréquentes par Cluster"):
        """
        Identifie et affiche les transitions (A -> B) les plus fréquentes pour chaque cluster,
        en ignorant les transitions vides (nan -> nan).

        Paramètres:
        ----------- 
        tca : TCA object
            L'objet TCA contenant les données de séquences.
        clusters : array-like
            Les labels de cluster pour chaque patient.
        n_top : int, optionnel
            Le nombre de transitions à afficher par cluster (défaut : 6).
        """

        df_seqs = tca.data.drop(columns=[tca.index_col])
        unique_clusters = np.unique(clusters)
        
        # Extraire toutes les transitions (paires d'états consécutifs)
        all_transitions = []
        for i in range(len(df_seqs.columns) - 1):
            t1, t2 = df_seqs.columns[i], df_seqs.columns[i+1]
            # On crée la paire sous forme de chaîne
            pair = df_seqs[t1].astype(str) + " → " + df_seqs[t2].astype(str)
            all_transitions.append(pair)
        
        df_trans = pd.concat(all_transitions, axis=1)
        
        # Calculer la fréquence de chaque transition par cluster
        trans_counts = []
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster)
            # Compte les occurrences
            counts = df_trans[cluster_mask].apply(pd.Series.value_counts).sum(axis=1)
            counts = counts / cluster_mask.sum() # Normalisation
            counts.name = f"Cluster {cluster}"
            trans_counts.append(counts)
            
        df_diff = pd.concat(trans_counts, axis=1).fillna(0)
        
        # Enlève "nan -> nan", et aussi "D -> nan" ou "nan -> C"
        df_diff = df_diff[~df_diff.index.str.contains('nan', case=False)]
        
        # Calculer le score de discrimination (Écart-type)
        df_diff['score'] = df_diff.std(axis=1)
        top_transitions = df_diff.sort_values(by='score', ascending=False).head(n_top)
        top_transitions = top_transitions.drop(columns=['score'])

        # Affichage
        if top_transitions.empty:
            print("Aucune transition valide trouvée après filtrage des NaNs.")
            return

        top_transitions.plot(kind='barh', figsize=(10, 8), width=0.8, edgecolor='black')
        plt.title(titre, fontsize=15, fontweight='bold')
        plt.xlabel("Fréquence moyenne par patient")
        plt.ylabel("Type de Transition")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.legend(title="Clusters")
        plt.tight_layout()
        plt.show()


def plot_extreme_trajectories(tca, clusters, distance_matrix, coverage_threshold=1.0, title="Trajectoires Médoïdes vs Extrêmes par cluster"):
    """
    Affiche la trajectoire du médoïde et celle de la trajectoire la plus éloignée pour chaque cluster.

    Paramètres:
    ------------
    tca : TCA object
        L'objet TCA contenant les données de séquences.
    clusters : array-like
        Les labels de cluster pour chaque patient.
    distance_matrix : ndarray
        La matrice de distance globale.
    coverage_threshold : float, optionnel
        Le seuil de tolérance requis par la fonction de calcul (par défaut 1.0).
    title : str
        Le titre du graphique.
    """
    unique_clusters = np.unique(clusters)
    df_seqs = tca.data.drop(columns=[tca.index_col])
    time_points = df_seqs.columns
    n_clusters = len(unique_clusters)

    # 1. On récupère les médoïdes
    rep_seq_results = extract_representative_sequences(df_seqs, distance_matrix, clusters, coverage_threshold)

    state_to_idx = {state: idx for idx, state in enumerate(tca.alphabet)}
    colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(tca.alphabet))]
    
    custom_cmap = mcolors.ListedColormap(colors)
    custom_cmap.set_bad(color='lightgray') 
    bounds = np.arange(len(tca.alphabet) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 1.5 * n_clusters), sharex=True)
    if n_clusters == 1: axes = [axes]

    for i, cluster_label in enumerate(unique_clusters):
        ax = axes[i]
        
        # Récupération de l'index du médoïde
        medoid_global_idx = rep_seq_results[cluster_label]['medoid_index']
        
        # Tous les index des patients de ce cluster précis
        cluster_indices = np.where(clusters == cluster_label)[0]
        
        # Recherche de l'extrême : on cherche le patient du cluster le plus éloigné de ce médoïde
        distances_to_medoid = distance_matrix[medoid_global_idx, cluster_indices]
        extreme_local_idx = np.argmax(distances_to_medoid)
        extreme_global_idx = cluster_indices[extreme_local_idx]

        seq_medoid = df_seqs.iloc[medoid_global_idx].map(state_to_idx).astype(float).values
        seq_extreme = df_seqs.iloc[extreme_global_idx].map(state_to_idx).astype(float).values
        
        # Superposition (Extrême en haut, Médoïde en bas)
        matrix = np.vstack([seq_extreme, seq_medoid])
        
        ax.imshow(matrix, aspect='auto', cmap=custom_cmap, norm=norm, interpolation='none')
        
        ax.set_title(f"Cluster {cluster_label} (n={len(cluster_indices)})", fontweight='bold', fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Extrême", "Médoïde"], fontsize=11, fontweight='bold')
        
        ax.axhline(0.5, color='white', linewidth=3)
        ax.tick_params(axis='y', length=0) 

    step = 3 
    axes[-1].set_xticks(np.arange(0, len(time_points), step))
    axes[-1].set_xticklabels(time_points[::step], rotation=45, ha='right')
    axes[-1].set_xlabel("Temps (Mois)", fontsize=12)

    patches = [mpatches.Patch(color=colors[i], label=tca.states[i]) for i in range(len(tca.alphabet))]
    patches.append(mpatches.Patch(color='lightgray', label="Pas de données"))
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(len(tca.alphabet)+1, 5), frameon=False, fontsize=11)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def summarize_clusters(tca, clusters):
        """
        Génère un résumé de chaque cluster.

        Paramètres:
        -----------
        tca : TCA object
        clusters : array-like   
            Les labels de cluster pour chaque patient.
        """
        unique_clusters = np.unique(clusters)
        df_seqs = tca.data.drop(columns=[tca.index_col])
        total_n = len(clusters)
        
        all_trans = []
        for i in range(len(df_seqs.columns) - 1):
            pair = df_seqs.iloc[:, i].astype(str) + " → " + df_seqs.iloc[:, i+1].astype(str)
            all_trans.append(pair)
        df_trans = pd.concat(all_trans, axis=1)

        summary_text = "========================================================\n"
        summary_text += "RESUME  DES CLUSTERS\n"
        summary_text += "========================================================\n\n"

        for cluster in unique_clusters:
            mask = (clusters == cluster)
            c_data = df_seqs[mask]
            n = mask.sum()
            perc = (n / total_n) * 100

            # --- A. Statistiques de Temps ---
            # Temps moyen passé dans chaque état
            # On compte les occurrences de chaque état et on divise par le nombre de patients
            counts = c_data.apply(pd.Series.value_counts).sum(axis=1).fillna(0)
            avg_durations = counts / n
            top_state = avg_durations.idxmax()
            top_duration = avg_durations.max()

            # --- B. Nombre de changements d'états ---
            changes = 0
            for i in range(len(c_data.columns) - 1):
                changes += ((c_data.iloc[:, i] != c_data.iloc[:, i+1]) & c_data.iloc[:, i].notna() & c_data.iloc[:, i+1].notna()).sum()
            avg_changes = changes / n

            # --- C. Transitions Signatures (Top 2) ---
            c_trans = df_trans[mask].apply(pd.Series.value_counts).sum(axis=1) / n
            c_trans = c_trans[~c_trans.index.str.contains('nan', case=False)]
            top_2_trans = c_trans.sort_values(ascending=False).head(2).index.tolist()

            # --- D. Rédaction du bloc ---
            summary_text += f"CLUSTER {cluster} : '{top_state.upper()}-DOMINANT'\n"
            summary_text += f"   • Taille       : {n} patients ({perc:.1f}% des patients)\n"
            summary_text += f"   • État Fréquent     : '{top_state}' (durée moyenne : {top_duration:.1f} mois)\n"
            summary_text += f"   • Transitions fréquentes   : {', '.join(top_2_trans)}\n"
            
            
            summary_text += "\n"

        summary_text += "========================================================"
        print(summary_text)
        return summary_text


def profil_clusters_heatmap(df_clinical, cluster_labels, variables_label=None, titre="Profil des Clusters - Heatmap desreprésentation"):
    """
    Génère une Heatmap montrant la sur/sous-représentation des 
    caractéristiques cliniques au sein de chaque cluster.

    Paramètres:
    ----------- 
    df_clinical : pd.DataFrame
        DataFrame contenant les variables cliniques (sans la colonne d'identifiant).
    cluster_labels : array-like
        Les labels de cluster pour chaque patient.
    titre : str, optionnel
        Le titre de la heatmap (défaut : "Profil des Clusters - Heatmap des représentation").
    """
    df = df_clinical.copy()

    if variables_label is not None:
        df = df.replace(variables_label)

    df['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
    
    lignes_ecarts = []
    
    for col in df_clinical.columns:
        # Répartition de cette variable dans la population globale (en %)
        rep_globale = df[col].value_counts(normalize=True) * 100
        
        # Répartition dans chaque cluster
        for cluster in df['Cluster'].unique():
            df_cluster = df[df['Cluster'] == cluster]
            rep_cluster = df_cluster[col].value_counts(normalize=True) * 100
            
            # Alignement avec l'index global (remplit avec 0% si une modalité est absente du cluster)
            rep_cluster = rep_cluster.reindex(rep_globale.index, fill_value=0)
            
            ecarts = rep_cluster - rep_globale
            
            for modalite, valeur in ecarts.items():
                lignes_ecarts.append({
                    'Caractéristique': f"{col.capitalize()} : {modalite}",
                    'Cluster': cluster,
                    'Ecart_Points': valeur
                })
                
    df_ecarts = pd.DataFrame(lignes_ecarts)
    
    # Lignes = Variables, Colonnes = Clusters
    matrice = df_ecarts.pivot(index='Caractéristique', columns='Cluster', values='Ecart_Points')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice, annot=True, cmap="bwr", center=0, fmt=".1f", 
                cbar_kws={'label': "Écart à la moyenne globale (en points de %)"})
    
    plt.title(titre, fontweight='bold', pad=20)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


