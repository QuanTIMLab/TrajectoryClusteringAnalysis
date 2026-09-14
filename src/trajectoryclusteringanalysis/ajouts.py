import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from lifelines import KaplanMeierFitter

# Convert from long to wide format
def long_to_wide_format(df, id_col, time_col, value_col, prefix="time_", static_cols=None, shift_time=False, sample_frac=0.1, random_state=42):
    """
    Convert a DataFrame from long to wide format.

    Parameters:
    df : pd.DataFrame
        Original long-format DataFrame.
    id_col : str
        Name of the column identifying patients/individuals (e.g. 'id_patient').
    time_col : str
        Name of the time column (e.g. 'month', 'time').
    value_col : str
        Name of the column containing states/values (e.g. 'care_status', 'etat').
    prefix : str, optional
        Prefix to add to the new time columns created (default: "time_").
    static_cols : list of str, optional
        Clinical variables to extract.
        Names of static columns to keep in the pivoted DataFrame.
    shift_time : bool, optional
        If True, adds +1 to the time value when renaming
        (useful if times start at 0 but you want to display M1, M2...).
    sample_frac : float, optional
        Fraction of individuals to keep (between 0.0 and 1.0).
        Default: 0.1 (keeps 10%).
    random_state : int, optional
        Random seed (default: 42).
        
    Returns:
    pd.DataFrame
        Pivoted wide-format DataFrame, with the identifier as the first column
        followed by time columns named with the prefix.
    """
    
    # Isolate unique IDs
    unique_ids = df[id_col].unique()
    
    # Compute the number of patients to keep
    n_samples = max(1, int(len(unique_ids) * sample_frac)) 
    
    rng = np.random.default_rng(random_state)
    sampled_ids = rng.choice(unique_ids, size=n_samples, replace=False)
    
    # Filter the long DataFrame for sampled patients
    df_sampled = df[df[id_col].isin(sampled_ids)]
    
    # Keep static columns
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
        
    # Reset index at the very end to return id_col as a normal column
    if static_cols is not None:
        return pivoted_data.reset_index(), df_clinical
    else:
        return pivoted_data.reset_index()
    

# Visualisation de la matrice de coûts
def plot_cost_matrix(cost_matrix, ax=None, title="Substitution Cost Matrix", cmap="YlOrRd", annot_fmt=".2f"):
    """
    Displays the cost matrix as a heatmap.

    Parameters:
    cost_matrix: pd.DataFrame
        The substitution cost matrix.
    ax: matplotlib.axes.Axes, optional
        The axis on which to plot the graph. If None, a new figure is created.
    title: str, optional
        The title of the graph.
    cmap: str or matplotlib.colors.Colormap, optional
        The color palette used for the heatmap (default: “YlOrRd”).
    annot_fmt: str, optional
        The formatting for the numbers displayed in the cells (default: “.2f” for 2 decimal places).

    Returns:
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        is_autonomous = True
    else:
        is_autonomous = False
        
    sns.heatmap(
        cost_matrix, 
        annot=True, 
        cmap=cmap, 
        cbar_kws={'label': 'Substitution cost'},
        fmt=annot_fmt,
        linewidths=0.5, 
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title(title, pad=15, fontsize=14, fontweight='bold')
    ax.set_xlabel("Final State", fontsize=12)
    ax.set_ylabel("Initial State", fontsize=12)
    
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    
    if is_autonomous:
        plt.tight_layout()
        plt.show()
        
    return ax


def plot_transversal_entropy(tca, clusters=None, title="Evolution of Transversal Entropy"):
        """
        Calculates and displays the transverse entropy (TraMineR's seqHtplot).
        If 'clusters' is provided, displays one curve per cluster.

        Parameters:
        tca: TCA object
            The TCA object containing the sequence data.
        clusters: array-like, optional
            The cluster labels for each patient. If None, displays the average for the entire dataset.
        """        
        df_seqs = tca.data.drop(columns=[tca.index_col])
        time_points = df_seqs.columns
        n_states = len(tca.alphabet)

        def calculate_entropy(df):
            entropies = []
            for col in df.columns:
                # Calculating the proportions p_i for each state
                probs = df[col].value_counts(normalize=True)
                # Shannon formula: -sum(p_i * log(p_i))
                # We normalize by log(n_states) to obtain a score between 0 and 1
                h = -np.sum(probs * np.log(probs)) / np.log(n_states) if n_states > 1 else 0
                entropies.append(h)
            return entropies

        plt.figure(figsize=(12, 6))

        if clusters is None:
            # Total Entropy
            overall_h = calculate_entropy(df_seqs)
            plt.plot(time_points, overall_h, label="Cohorte Totale", color='black', linewidth=3)
        else:
            # One curve per cluster
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                cluster_data = df_seqs[clusters == cluster]
                cluster_h = calculate_entropy(cluster_data)
                plt.plot(time_points, cluster_h, label=f"Cluster {cluster}")

        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Entropy Index (Normalized)")
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title="Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_mean_time(tca, clusters=None, title="Average Time Spent in Each State"):
        """
        Calculates and displays the average time spent in each state (seqmtplot).

        Parameters:
        tca: TCA object
            The TCA object containing the sequence data.
        clusters: array-like, optional
            The cluster labels for each patient. If None, displays the average for the entire dataset.
        title: str, optional
            The title of the plot.
        """
        df_seqs = tca.data.drop(columns=[tca.index_col])
        
        def get_means(df):
            if df.empty:
                return pd.Series(0, index=tca.alphabet)
                
            counts = df.apply(pd.Series.value_counts, dropna=False, axis=1).fillna(0)
            for state in tca.alphabet:
                if state not in counts.columns:
                    counts[state] = 0
            
            return counts[tca.alphabet].mean().fillna(0)

        plt.figure(figsize=(10, 6))
        
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(tca.alphabet))]

        if clusters is None:
            means = get_means(df_seqs)
            bars = plt.bar(tca.states, means, color=colors, edgecolor='black', alpha=0.8)
            plt.ylabel("Nombre moyen de mois")
        else:
            # Cluster Comparison (Grouped Bars)
            unique_clusters = np.unique(np.array(clusters))
            n_clusters = len(unique_clusters)
            width = 0.8 / n_clusters
            x = np.arange(len(tca.states))

            for i, cluster in enumerate(unique_clusters):
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
    Identifies the medoid sequence for each cluster and calculates its representativeness.

    Parameters:
    df_sequences: DataFrame
        The DataFrame containing only the sequences (without the patient index column).
    distance_matrix: ndarray
        The distance matrix.
    cluster_labels: array-like
        The cluster labels assigned to each patient (size N).
    coverage_threshold: float
        The maximum acceptable distance for a patient to be considered “represented” by the medoid.

    Returns:
    dict: A dictionary containing the medoid information for each cluster.
    """
    unique_clusters = np.unique(cluster_labels)
    results = {}
    
    for cluster in unique_clusters:
        idx_cluster = np.where(cluster_labels == cluster)[0]
        n_patients = len(idx_cluster)
        
        # Extract the distance submatrix for this cluster only
        sub_matrix = distance_matrix[np.ix_(idx_cluster, idx_cluster)]
        
        # Find the medoid (the one whose sum of distances to the others is minimal)
        sum_distances = sub_matrix.sum(axis=1)
        medoid_relative_idx = np.argmin(sum_distances) # Index dans la sous-matrice
        medoid_absolute_idx = idx_cluster[medoid_relative_idx] # Index dans le jeu de données global
        
        # Calculate the coverage rate
        distances_to_medoid = sub_matrix[medoid_relative_idx]
        covered_count = np.sum(distances_to_medoid <= coverage_threshold)
        coverage_rate = (covered_count / n_patients) * 100
        
        # Save the results
        results[cluster] = {
            'medoid_index': medoid_absolute_idx,
            'sequence': df_sequences.iloc[medoid_absolute_idx].values,
            'n_patients': n_patients,
            'coverage_rate': coverage_rate,
            'covered_count': covered_count
        }
        
    return results


def plot_representative_sequences(tca, rep_seq_results, title="Representative Sequences (Medoids)"):
    """
    Displays representative sequences along with their coverage rates.

    Parameters:
    tca: TCA object
    rep_seq_results: dict
        A dictionary containing information about the representative sequences for each cluster.
    title: str
        The title of the graph.
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
        
        # Convert the sequence to digital format for display
        seq_numeric = np.array([state_to_idx.get(s, np.nan) for s in info['sequence']])
        matrix = seq_numeric.reshape(1, -1).astype(float)
        
        ax.imshow(matrix, aspect='auto', cmap=custom_cmap, norm=norm, interpolation='none')
        
        # Y-axis labels: Cluster name and coverage
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
    Calculates the medoid coverage rate (given by k-medoids).

    Parameters:
    df_sequences: DataFrame
        The DataFrame containing only the sequences.
    distance_matrix: ndarray
        The global distance matrix.
    cluster_labels: array-like
        The cluster labels assigned to each patient.
    medoids_indices: array-like
        The absolute indices of the medoids (typically kmedoids_model.medoid_indices_).
    coverage_threshold: float
        The maximum acceptable distance for the neighborhood.

    Returns:
    dict: A dictionary containing the medoid information for each cluster.
    """
    unique_clusters = np.unique(cluster_labels)
    results = {}
    
    # Creating a dictionary to associate each medoid with its cluster
    medoid_dict = {cluster_labels[idx]: idx for idx in medoids_indices}
    
    for cluster in unique_clusters:
        # Find all patients in this cluster
        idx_cluster = np.where(cluster_labels == cluster)[0]
        n_patients = len(idx_cluster)
        
        # Retrieve the medoid index for this cluster
        medoid_absolute_idx = medoid_dict[cluster]
        
        # Directly extract the distances between this medoid and the patients in its cluster
        distances_to_medoid = distance_matrix[medoid_absolute_idx, idx_cluster]
        
        # Calculate the coverage rate
        covered_count = np.sum(distances_to_medoid <= coverage_threshold)
        coverage_rate = (covered_count / n_patients) * 100
        
        # Save the results
        results[cluster] = {
            'medoid_index': medoid_absolute_idx,
            'sequence': df_sequences.iloc[medoid_absolute_idx].values,
            'n_patients': n_patients,
            'coverage_rate': coverage_rate,
            'covered_count': covered_count
        }
        
    return results


def plot_trajectory_decision_tree(df_clinical, cluster_labels, max_depth=3, variables_label=None, titre="Clinical Pathway Decision Tree"):
    """
    Trains and displays a simplified decision tree to see which variables
    lead to the different clusters.

    Parameters:
    df_clinical: pd.DataFrame
        A DataFrame containing the clinical variables (excluding the ID column).
    cluster_labels: array-like
        The cluster labels for each patient.
    max_depth: int, optional
        The maximum depth of the tree (default: 3).
    variables_label: dict, optional
        A dictionary to replace variable values (default: None).
    title: str, optional
        The title of the plot (default: “Clinical Trajectory Decision Tree”).
    """
    X = df_clinical.copy()
    
    if variables_label is not None:
        X = X.replace(variables_label)
        
    X = pd.get_dummies(X)
        
    y = cluster_labels
    
    # Training the decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=0.05)
    clf.fit(X, y)
    
    # Visualization
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
        lines = text.split('\n')
        lines_new = []
        
        for line in lines:
            if "samples" in line or "value" in line:
                continue
                
            if "<=" in line:
                col_name = line.split(" <= ")[0]
                
                if "_" in col_name and "0.5" in line:
                    var_name, modalite = col_name.split("_", 1)
                    lines_new.append(f"{var_name} : {modalite} ?")
                    lines_new.append("<- No  | Yes ->")
                else:
                    lines_new.append(f"{line} ?")
            else:
                lines_new.append(line)
                
        new_text = '\n'.join(lines_new)
        text_obj.set_text(new_text)
        text_obj.set_fontweight('bold')

    plt.title(titre, fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return clf


def plot_time_to_event(df_sequences, cluster_labels, target_state='S', time_unit="Mois", titre="Time to reach the target state by cluster"):
    """
    Displays curves to compare the rate at which each cluster reaches the target state.

    Parameters:
    df_sequences: pd.DataFrame
        DataFrame containing the state sequences (without the ID column).
    cluster_labels: array-like
        The cluster labels for each patient.
    target_state: str, optional
        The state of interest for the survival analysis (default: ‘S’).
    time_unit: str, optional
        The time unit to display on the X-axis (default: “Months”).
    """
    T = [] # Time until the event (or maximum follow-up time if not reached)
    E = [] # 1 if target state is reached, 0 otherwise
    
    max_time = df_sequences.shape[1]
    
    # Calculating arrival times for each patient
    for i in range(len(df_sequences)):
        seq = df_sequences.iloc[i].values
        # First occurrence of the target state
        indices = np.where(seq == target_state)[0]
        
        if len(indices) > 0:
            T.append(indices[0]) # Time is the index of the first occurrence
            E.append(1)          # The event occurred
        else:
            T.append(max_time)   # Never reached that stage
            E.append(0)          # The event did not occur
            
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        # Filter patients belonging to this cluster
        idx = (cluster_labels == cluster)
        T_cluster = np.array(T)[idx]
        E_cluster = np.array(E)[idx]
        
        kmf.fit(T_cluster, event_observed=E_cluster, label=f"Cluster {cluster}")
        
        # plot_cumulative_density displays the proportion that has REACHED the state (rises from 0 to 1)
        kmf.plot_cumulative_density(ci_show=False, linewidth=2.5)

    plt.title(titre, fontsize=15, pad=15, fontweight='bold')
    plt.xlabel(f"Temps écoulé ({time_unit})", fontsize=12)
    plt.ylabel(f"Proportion de patients dans l'état '{target_state}'", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
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


def plot_cluster_profiles(df_clinical, cluster_labels, variables_label=None, variables_to_plot=None, titre="Clinical Profile of Clusters"):
    """
    Displays the complete clinical profile as horizontal stacked bars.

    Parameters:
    df_clinical: pd.DataFrame
        A DataFrame containing the clinical variables (excluding the ID column).
    cluster_labels: array-like
        The cluster labels for each patient.
    variables_label: dict, optional
        A dictionary to replace numerical values with their actual names.
    variables_to_plot: list, optional
        A list of specific columns to include in the plot. If None, all columns will be used.
    title: str, optional
        The title of the plot.
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
            
            if width > 3:
                col_idx = int(np.floor(ax.patches.index(patch) / len(variables_to_plot)))
                if col_idx < len(df_plot.columns):
                    label_text = df_plot.columns[col_idx]
                    
                    ax.text(
                        x + width/2, 
                        y + height/2, 
                        f"{label_text}\n({width:.1f}%)", 
                        ha='center', 
                        va='center', 
                        color='black',      
                        fontweight='bold',
                        fontsize=9         
                    )
        
        ax.set_title(f"Patient Profile : Cluster {cluster} (n = {len(df_c)} patients)", fontsize=13, pad=10)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    axes[-1].set_xlabel("Proportion within the cluster (%)", fontsize=11)
    plt.tight_layout()
    plt.suptitle(titre, fontsize=14, fontweight='bold', y=1.01)
    plt.show()


def plot_frequent_transitions(tca, clusters, n_top=6, titre="Most Frequent Transitions by Cluster"):
        """
        Identifies and displays the most frequent transitions (A -> B) for each cluster,
        ignoring empty transitions (nan -> nan).

        Parameters:
        -----------
        tca: TCA object
            The TCA object containing the sequence data.
        clusters: array-like
            The cluster labels for each patient.
        n_top: int, optional
            The number of transitions to display per cluster (default: 6).
        """

        df_seqs = tca.data.drop(columns=[tca.index_col])
        unique_clusters = np.unique(clusters)
        
        # Extract all transitions (pairs of consecutive states)
        all_transitions = []
        for i in range(len(df_seqs.columns) - 1):
            t1, t2 = df_seqs.columns[i], df_seqs.columns[i+1]
            # We create the pair as a string
            pair = df_seqs[t1].astype(str) + " → " + df_seqs[t2].astype(str)
            all_transitions.append(pair)
        
        df_trans = pd.concat(all_transitions, axis=1)
        
        # Calculate the frequency of each transition by cluster
        trans_counts = []
        for cluster in unique_clusters:
            cluster_mask = (clusters == cluster)
            # Count Occurrences
            counts = df_trans[cluster_mask].apply(pd.Series.value_counts).sum(axis=1)
            counts = counts / cluster_mask.sum() # Normalization
            counts.name = f"Cluster {cluster}"
            trans_counts.append(counts)
            
        df_diff = pd.concat(trans_counts, axis=1).fillna(0)
        
        # Remove “nan -> nan,” as well as “D -> nan and “nan -> C”
        df_diff = df_diff[~df_diff.index.str.contains('nan', case=False)]
        
        # Calculate the Standard Deviation
        df_diff['score'] = df_diff.std(axis=1)
        top_transitions = df_diff.sort_values(by='score', ascending=False).head(n_top)
        top_transitions = top_transitions.drop(columns=['score'])

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


def plot_extreme_trajectories(tca, clusters, distance_matrix, coverage_threshold=1.0, title="Medoid Trajectories vs. Extremes by Cluster"):
    """
    Displays the medoid trajectory and the trajectory of the farthest point for each cluster.

    Parameters:
    ------------
    tca: TCA object
        The TCA object containing the sequence data.
    clusters: array-like
        The cluster labels for each patient.
    distance_matrix: ndarray
        The global distance matrix.
    coverage_threshold: float, optional
        The tolerance threshold required by the calculation function (default 1.0).
    title: str
        The title of the graph.
    """
    unique_clusters = np.unique(clusters)
    df_seqs = tca.data.drop(columns=[tca.index_col])
    time_points = df_seqs.columns
    n_clusters = len(unique_clusters)

    # We retrieve the medoids
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
        
        # Médoïde index
        medoid_global_idx = rep_seq_results[cluster_label]['medoid_index']
        
        cluster_indices = np.where(clusters == cluster_label)[0]
        
        # Searching for the extreme: We are looking for the patient in the cluster farthest from this medoid
        distances_to_medoid = distance_matrix[medoid_global_idx, cluster_indices]
        extreme_local_idx = np.argmax(distances_to_medoid)
        extreme_global_idx = cluster_indices[extreme_local_idx]

        seq_medoid = df_seqs.iloc[medoid_global_idx].map(state_to_idx).astype(float).values
        seq_extreme = df_seqs.iloc[extreme_global_idx].map(state_to_idx).astype(float).values
        
        matrix = np.vstack([seq_extreme, seq_medoid])
        
        ax.imshow(matrix, aspect='auto', cmap=custom_cmap, norm=norm, interpolation='none')
        
        ax.set_title(f"Cluster {cluster_label} (n={len(cluster_indices)})", fontweight='bold', fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Extreme", "Medoid"], fontsize=11, fontweight='bold')
        
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
    Generates a summary for each cluster.

    Parameters:
    tca: TCA object
    clusters: array-like
        The cluster labels for each patient.
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
    summary_text += "SUMMARY OF CLUSTERS\n"
    summary_text += "========================================================\n\n"

    for cluster in unique_clusters:
        mask = (clusters == cluster)
        c_data = df_seqs[mask]
        n = mask.sum()
        perc = (n / total_n) * 100

        # A. Time Statistics
        # Average time spent in each state
        # We count the number of occurrences of each state and divide by the number of patients
        counts = c_data.apply(pd.Series.value_counts).sum(axis=1).fillna(0)
        avg_durations = counts / n
        top_state = avg_durations.idxmax()
        top_duration = avg_durations.max()

        # B. Number of state changes
        changes = 0
        for i in range(len(c_data.columns) - 1):
            changes += ((c_data.iloc[:, i] != c_data.iloc[:, i+1]) & c_data.iloc[:, i].notna() & c_data.iloc[:, i+1].notna()).sum()
        avg_changes = changes / n

        # C. Most Common Transitions (Top 2)
        c_trans = df_trans[mask].apply(pd.Series.value_counts).sum(axis=1) / n
        c_trans = c_trans[~c_trans.index.str.contains('nan', case=False)]
        top_2_trans = c_trans.sort_values(ascending=False).head(2).index.tolist()

        summary_text += f"CLUSTER {cluster} : '{top_state.upper()}-DOMINANT'\n"
        summary_text += f"   • Size       : {n} patients ({perc:.1f}% of patients)\n"
        summary_text += f"   • Frequent state     : '{top_state}' (mean duration : {top_duration:.1f})\n"
        summary_text += f"   • Frequent transition   : {', '.join(top_2_trans)}\n"
            
            
        summary_text += "\n"

    summary_text += "========================================================"
    print(summary_text)
    return summary_text


def profil_clusters_heatmap(df_clinical, cluster_labels, variables_label=None, titre="Cluster Profiles - Heatmap of Representations"):
    """
    Generates a heatmap showing the over- or under-representation of
    clinical features within each cluster.

    Parameters:
    df_clinical: pd.DataFrame
        DataFrame containing the clinical variables (excluding the ID column).
    cluster_labels: array-like
        The cluster labels for each patient.
    title: str, optional
        The title of the heatmap (default: “Cluster Profiles - Representation Heatmap”).
    """
    df = df_clinical.copy()

    if variables_label is not None:
        df = df.replace(variables_label)

    df['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
    
    lignes_ecarts = []
    
    for col in df_clinical.columns:
        # Distribution of this variable in the overall population (in %)
        rep_globale = df[col].value_counts(normalize=True) * 100
        
        # Distribution within each cluster
        for cluster in df['Cluster'].unique():
            df_cluster = df[df['Cluster'] == cluster]
            rep_cluster = df_cluster[col].value_counts(normalize=True) * 100
            
            rep_cluster = rep_cluster.reindex(rep_globale.index, fill_value=0)
            
            ecarts = rep_cluster - rep_globale
            
            for modalite, valeur in ecarts.items():
                lignes_ecarts.append({
                    'Caractéristique': f"{col.capitalize()} : {modalite}",
                    'Cluster': cluster,
                    'Ecart_Points': valeur
                })
                
    df_ecarts = pd.DataFrame(lignes_ecarts)
    
    matrice = df_ecarts.pivot(index='Caractéristique', columns='Cluster', values='Ecart_Points')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice, annot=True, cmap="bwr", center=0, fmt=".1f", 
                cbar_kws={'label': "Écart à la moyenne globale (en points de %)"})
    
    plt.title(titre, fontweight='bold', pad=20)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


