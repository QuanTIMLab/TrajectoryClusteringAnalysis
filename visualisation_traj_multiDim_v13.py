# Standard library imports
import logging
from typing import Dict, Generator, List, NamedTuple, TypedDict, Union

# Third-party imports
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, dash_table, html, State
from plotly.subplots import make_subplots
from functools import lru_cache

# Local imports
from config import CONFIG

# Initiialization of a pandas DataFrame (used to store and manipulate data within the application)
df: pd.DataFrame = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientData(NamedTuple):
    patient_id: int
    treatments: Dict[str, List[tuple]]

class HeatmapData(TypedDict):
    x: List[int]
    y: List[int]
    z: np.ndarray

class SampleButtonConfig(TypedDict):
    label: str
    method: str
    args: List[Dict[str, List[bool]]]

class HeatmapButtonData(TypedDict):
    label: str
    method: str
    args: List[Dict]

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout avec filtres pour dataset
app.layout = html.Div([
    dcc.Store(id='sample-size-store', data=CONFIG['SAMPLE_SIZES'][0]),
    dcc.Tabs([
        dcc.Tab(label='Visualization', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Taille de l'échantillon"),
                        dcc.Dropdown(
                            id='vis-sample-size-menu',
                            options=[{"label": "Tous", "value": "Tous"}] + 
                                [{"label": str(size), "value": size} for size in CONFIG['SAMPLE_SIZES']],
                            value="Tous",
                            clearable=False,
                            style={'width': '100%'}
                        )
                    ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("Traitement"),
                        dcc.Dropdown(
                            id='treatment-filter',
                            options=[{"label": "Tous", "value": "Tous"}] + 
                                [{"label": trt, "value": trt} for trt in CONFIG['TREATMENTS']],
                            value="Tous",
                            clearable=False,
                            style={'width': '100%'}
                        )
                    ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("Semaine de début (min-max)"),
                        dcc.RangeSlider(
                            id='start-week-filter',
                            min=0,
                            max=CONFIG['MAX_WEEK'],
                            step=1,
                            value=[0, CONFIG['MAX_WEEK']],
                            marks={i: str(i) for i in range(0, CONFIG['MAX_WEEK']+1, 5)}
                        )
                    ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("Semaine de fin (min-max)"),
                        dcc.RangeSlider(
                            id='end-week-filter',
                            min=0,
                            max=CONFIG['MAX_WEEK'],
                            step=1,
                            value=[0, CONFIG['MAX_WEEK']],
                            marks={i: str(i) for i in range(0, CONFIG['MAX_WEEK']+1, 5)}
                        )
                    ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Div([
                        html.Label("IDs patients (séparés par des virgules)"),
                        dcc.Input(
                            id='patient-ids-filter',
                            type='text',
                            placeholder='ex: 1,5,10-15',
                            style={'width': '100%'}
                        )
                    ], style={'width': '20%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    html.Button('Appliquer les filtres', id='apply-filters', n_clicks=0)
                ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                
                dcc.Graph(
                    id='main-graph',
                    style={'height': '200vh', 'width': '100%'} 
                )
            ], style={'display': 'inline-block', 'width': '100%'})
        ]),
        dcc.Tab(label='Dataset', children=[
            html.Div([
                html.H3("Selected Patients Data"),
                dash_table.DataTable(
                    id='patient-data-table',
                    style_table={'height': '85vh', 'overflowY': 'auto'},
                    page_size=20
                )
            ], style={'padding': '20px'})
        ])
    ])
], style={'height': '100vh', 'width': '100%'})

@app.callback(
    [Output('main-graph', 'figure'),
    Output('patient-data-table', 'data'),
    Output('patient-data-table', 'columns')],
    [Input('apply-filters', 'n_clicks')],
    [State('vis-sample-size-menu', 'value'),
    State('treatment-filter', 'value'),
    State('start-week-filter', 'value'),
    State('end-week-filter', 'value'),
    State('patient-ids-filter', 'value')]
)
def update_content_with_filters(n_clicks, sample_size, treatment, start_week_range, end_week_range, patient_ids_str):
    try:
        # Déterminer la taille de l'échantillon
        if sample_size == "Tous":
            actual_size = len(df['Patient_ID'].unique())
        else:
            actual_size = int(sample_size)
        
        # Obtenir l'échantillon de base
        filtered_df = _get_patient_sample(actual_size)
        
        # Appliquer les filtres
        # 1. Filtre par traitement
        if treatment != "Tous":
            filtered_df = filtered_df[filtered_df['Treatment'] == treatment]
        
        # 2. Filtre par semaine de début
        filtered_df = filtered_df[(filtered_df['Start_Week'] >= start_week_range[0]) & 
                                (filtered_df['Start_Week'] <= start_week_range[1])]
        
        # 3. Filtre par semaine de fin
        filtered_df = filtered_df[(filtered_df['End_Week'] >= end_week_range[0]) & 
                                (filtered_df['End_Week'] <= end_week_range[1])]
        
        # 4. Filtre par IDs patients
        available_ids = df['Patient_ID'].unique()
        if patient_ids_str:
            patient_ids = parse_patient_ids(patient_ids_str, available_ids)
            filtered_df = filtered_df[filtered_df['Patient_ID'].isin(patient_ids)]
        
        # Vérifier qu'il reste des données après filtrage
        if filtered_df.empty:
            return {}, [], []
        
        # Créer la figure avec le DataFrame filtré
        fig = create_base_figure(filtered_df, CONFIG['TREATMENTS'])
        
        # Ajouter les glyphes
        _add_patient_traces(fig, filtered_df)
        
        # Mettre à jour la mise en page
        update_layout(fig, filtered_df, CONFIG['TREATMENTS'])
        
        # Préparer les données pour le tableau
        columns = [{"name": col, "id": col} for col in filtered_df.columns]
        
        return fig, filtered_df.to_dict('records'), columns
    except Exception as e:
        print(f"Error in update_content_with_filters: {e}")
        raise

# Initiialisation valeurs filters
@app.callback(
    [Output('start-week-filter', 'min'),
    Output('start-week-filter', 'max'),
    Output('start-week-filter', 'value'),
    Output('start-week-filter', 'marks'),
    Output('end-week-filter', 'min'),
    Output('end-week-filter', 'max'),
    Output('end-week-filter', 'value'),
    Output('end-week-filter', 'marks')],
    [Input('vis-sample-size-menu', 'value')]
)

def update_filter_ranges(sample_size):
    # Déterminer la plage des semaines en fonction de l'échantillon
    if sample_size == "Tous":
        sample_df = df
    else:
        actual_size = int(sample_size)
        sample_df = _get_patient_sample(actual_size)
    
    min_start = int(sample_df['Start_Week'].min())
    max_start = int(sample_df['Start_Week'].max())
    
    min_end = int(sample_df['End_Week'].min())
    max_end = int(sample_df['End_Week'].max())
    
    # Créer les marques pour les sliders (montrer une marque toutes les 5 semaines)
    start_marks = {i: str(i) for i in range(min_start, max_start+1, 5)}
    end_marks = {i: str(i) for i in range(min_end, max_end+1, 5)}
    
    return (min_start, max_start, [min_start, max_start], start_marks,
            min_end, max_end, [min_end, max_end], end_marks)

@app.callback(
    Output('sample-size-store', 'data'),
    [Input('vis-sample-size-menu', 'value')]
)
def update_sample_size(value):
    return value

# Core Data Generation
def generate_data(n_patients: int, max_week: int, treatments: List[str]) -> pd.DataFrame:
    """Generate patient treatment data with multiple possible treatment periods.
    
    Args:
        n_patients: Number of patients to generate
        max_week: Maximum number of weeks for treatment periods  
        treatments: List of available treatments

    Returns:
        DataFrame with columns: Patient_ID, Treatment, Start_Week, End_Week
    """
    data = []
    np.random.seed(42)
    
    for pid in range(1, n_patients + 1):
        treatment_periods = {trt: [] for trt in treatments}
        
        for start_week in range(max_week):
            n_trt = np.random.randint(1, 5)
            if n_trt > 0 and start_week < max_week - 1:
                trts = np.random.choice(treatments, n_trt, replace=False)
                for trt in trts:
                    end_week = np.random.randint(start_week + 1, max_week + 1)
                    if not any(start <= end_week and end >= start_week 
                            for start, end in treatment_periods[trt]):
                        treatment_periods[trt].append((start_week, end_week))
                        data.append({
                            'Patient_ID': pid,
                            'Treatment': trt,
                            'Start_Week': start_week,
                            'End_Week': end_week
                        })
    
    return pd.DataFrame(data)

def parse_patient_ids(id_string, available_ids):
    """
    Parse une chaîne d'IDs patients comme "1,3,5-10,15"
    
    Args:
        id_string: Chaîne à parser
        available_ids: Liste des IDs disponibles
    
    Returns:
        Liste d'IDs patients
    """
    if not id_string or id_string.strip() == '':
        return available_ids
        
    result = []
    parts = id_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Traiter les plages (ex: 5-10)
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            # Traiter les valeurs uniques
            try:
                result.append(int(part))
            except ValueError:
                # Ignorer les valeurs non entières
                pass
    
    # Ne garder que les IDs qui existent vraiment
    return [pid for pid in result if pid in available_ids]

# Data Processing
@lru_cache(maxsize=128)
def get_heatmap_data(df_hash: str, treatment: str, mode: str) -> Dict[str, Union[List[int], np.ndarray]]:
    """
    Génère les données de carte thermique pour la visualisation des traitements.
    
    Args:
        df_hash: Hash du DataFrame
        treatment: Nom du traitement
        mode: 'new' ou 'active'
    
    Returns:
        Dict avec plages de semaines x/y et matrice z
    """
    _validate_mode(mode)
    df = cache_store.get(df_hash)
    if df is None:
        raise ValueError(f"DataFrame avec hash {df_hash} non trouvé dans le cache")
    
    weeks_range = _get_week_range(df)
    data = _calculate_heatmap(df, treatment, weeks_range, mode)
    return {
        'x': weeks_range,
        'y': weeks_range,
        'z': data
    }

def _validate_mode(mode: str) -> None:
    if mode not in ['new', 'active']:
        raise ValueError("Mode must be 'new' or 'active'")

def _get_week_range(df: pd.DataFrame) -> List[int]:
    max_week = int(max(df['End_Week'].max(), df['Start_Week'].max()))
    return list(range(max_week + 1))  # Commencer à 0

def _calculate_heatmap(df: pd.DataFrame, treatment: str, weeks_range: List[int], mode: str = 'new') -> np.ndarray:
    data = np.zeros((len(weeks_range), len(weeks_range)))
    trt_df = df[df['Treatment'] == treatment]
    
    if mode == 'new':
        # Compter les patients uniques pour chaque combinaison début/fin
        for start_week in weeks_range:
            for end_week in weeks_range:
                if start_week <= end_week:  
                    patients = trt_df[
                        (trt_df['Start_Week'] == start_week) & 
                        (trt_df['End_Week'] == end_week)
                    ]['Patient_ID'].nunique()
                    data[start_week][end_week] = patients
    else:  # mode == 'active'
        for start_week in weeks_range:
            for end_week in weeks_range:
                if start_week <= end_week:  
                    active_patients = trt_df[
                        (trt_df['Start_Week'] <= start_week) & 
                        (trt_df['End_Week'] >= end_week)
                    ]['Patient_ID'].nunique()
                    
                    data[start_week][end_week] = active_patients
            
    return data

# Global cache store for DataFrames
cache_store = {}

def get_df_hash(df: pd.DataFrame) -> str:
    """Génère un hash unique pour le DataFrame"""
    df_hash = str(hash(pd.util.hash_pandas_object(df).sum()))
    cache_store[df_hash] = df
    return df_hash

# Visualization Components
def create_base_figure(df: pd.DataFrame, treatments: List[str]) -> go.Figure:
    fig = make_subplots(
        rows=len(treatments) + 2,
        cols=1, 
        row_heights=[0.15] * len(treatments) + [0.05, 0.35],
        vertical_spacing=0.02,
        specs=[[{"type": "heatmap"}] for _ in range(len(treatments))] + 
        [[{"type": "scatter"}], [{"type": "scatter"}]]
    )
    
    weeks_range = _get_week_range(df)
    df_hash = get_df_hash(df)  # Stocke le DataFrame dans le cache
    
    for i, trt in enumerate(treatments, 1):
        heatmap_data = get_heatmap_data(df_hash, trt, 'new')
        fig.add_trace(
            go.Heatmap(
                x=weeks_range,
                y=weeks_range,
                z=heatmap_data['z'],
                colorscale=CONFIG['COLORSCALE'],
                name=trt,
                coloraxis="coloraxis1",
                showscale=(i == 1),
                hovertemplate="Début: sem. %{y}<br>Fin: sem. %{x}<br>Nombre: %{z}<extra></extra>",
                zmin=0,
                zmid=df['Patient_ID'].nunique() / 10
            ),
            row=i, col=1
        )
        
        if i == len(treatments):
            fig.update_xaxes(title_text="Semaine de fin", row=i, col=1)
        else:
            fig.update_xaxes(title_text="", row=i, col=1)
        fig.update_yaxes(title_text=f"{trt}", row=i, col=1)
    
    #fig.update_yaxes(title_text="<br>Semaine de début", standoff=40)
    fig.add_annotation(
        text="Semaine de début",
        x=-0.03,
        y=0.75,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
        textangle=-90
    )
    fig.update_layout(coloraxis=dict(colorscale=CONFIG['COLORSCALE'], colorbar=dict(title="Patients")))
    
    return fig

def create_patient_traces(patient_df: pd.DataFrame, pid: int) -> List[go.Scatter]:
    """
    Creates visualization traces for a patient's treatment timeline.
    
    Args:
        patient_df: DataFrame containing patient's treatment data
        pid: Patient identifier
    
    Returns:
        List of Plotly Scatter traces for timeline visualization
    """
    traces = []
    
    # Calculate timeline range
    start_week = int(patient_df['Start_Week'].min())
    end_week = int(patient_df['End_Week'].max())
    weeks_range = list(range(start_week, end_week + 1))
    
    # Add baseline timeline
    traces.append(_create_timeline_trace(weeks_range, pid))
    
    # Add treatment markers
    treatment_traces = _create_treatment_markers(patient_df, weeks_range, pid)
    traces.extend(treatment_traces)
    
    return traces

def _create_timeline_trace(weeks_range: List[int], pid: int) -> go.Scatter:
    """Creates the baseline timeline trace."""
    return go.Scatter(
        x=weeks_range,
        y=[pid] * len(weeks_range),
        mode='lines',
        line=dict(color='lightgray', width=0.5, dash='dot'),
        showlegend=False
    )

def _create_treatment_markers(patient_df: pd.DataFrame, weeks_range: List[int], pid: int) -> List[go.Scatter]:
    traces = []
    
    for week in weeks_range:
        # Trouver les traitements actifs pour cette semaine
        active_treatments = patient_df[
            (patient_df['Start_Week'] <= week) &
            (patient_df['End_Week'] >= week)
        ]
        
        n_active_trts = len(active_treatments)
        if n_active_trts > 0:
            trt_names = ', '.join(active_treatments['Treatment'])
            traces.append(go.Scatter(
                x=[week],
                y=[pid],
                mode='markers',
                marker=dict(
                    symbol=CONFIG['SYMBOLS'].get(n_active_trts, 'circle'),
                    size=CONFIG['GLYPH_BASE_SIZE'] + CONFIG['GLYPH_SIZE_INCREMENT'] * n_active_trts,
                    color=CONFIG['GLYPH_COLORS'].get(n_active_trts, '#808080'),
                    line=dict(color='white', width=1)
                ),
                hovertemplate=f"Patient {pid}<br>Semaine {week}<br>Traitements: {trt_names}<extra></extra>",
                showlegend=False
            ))
    
    return traces

def create_heatmap_buttons(df: pd.DataFrame, treatments: List[str]) -> List[HeatmapButtonData]:
    """
    Creates buttons for switching between heatmap visualization types.
    
    Args:
        df: DataFrame containing patient treatment data
        treatments: List of treatment names
    
    Returns:
        List of button configurations for heatmap type switching
    """
    cached_data = _calculate_heatmap_data(df, treatments)
    return [
        _create_new_patients_button(cached_data, treatments),
        _create_active_patients_button(cached_data, treatments)
    ]

def _calculate_heatmap_data(df: pd.DataFrame, treatments: List[str]) -> Dict:
    df_hash = get_df_hash(df)
    return {
        'new': {trt: get_heatmap_data(df_hash, trt, 'new') for trt in treatments},
        'active': {trt: get_heatmap_data(df_hash, trt, 'active') for trt in treatments}
    }

def _create_new_patients_button(cached_data: Dict, treatments: List[str]) -> HeatmapButtonData:
    """Creates button configuration for new patients visualization."""
    return {
        'label': "Nouveaux patients uniques par semaine",
        'method': "update",
        'args': [{
            # Modifier uniquement les traces de heatmap (qui sont aux indices 0 à len(treatments)-1)
            'z': [cached_data['new'][trt]['z'] for trt in treatments],
            'hovertemplate': [
                "Début: sem. %{y}<br>Fin: sem. %{x}<br>Nombre de patients uniques: %{z}<extra></extra>"
            ] * len(treatments)
        }, {}, list(range(len(treatments)))]  # Troisième argument: indices des traces à modifier
    }

def _create_active_patients_button(cached_data: Dict, treatments: List[str]) -> HeatmapButtonData:
    """Creates button configuration for active patients visualization."""
    return {
        'label': "Patients actifs",
        'method': "update",
        'args': [{
            # Modifier uniquement les traces de heatmap (qui sont aux indices 0 à len(treatments)-1)
            'z': [cached_data['active'][trt]['z'] for trt in treatments],
            'hovertemplate': [
                "Semaine: %{y}<br>Nombre de patients actifs: %{z}<extra></extra>"
            ] * len(treatments)
        }, {}, list(range(len(treatments)))]  # Troisième argument: indices des traces à modifier
    }

# Layout & Updates
def update_layout(fig: go.Figure, df: pd.DataFrame, treatments: List[str]) -> None:
    fig.update_layout(
        height=CONFIG['FIGURE_HEIGHT'],
        title={
            'text': "Visualisation des Trajectoires Thérapeutiques",
            'y': 0.95,  
            'x': 0.5,  
            'xanchor': 'center',  
            'yanchor': 'top',  
            'font': {
                'size': 24,
                'weight': 'bold' 
            }
        },
        margin=dict(l=50, r=50, t=100, b=50),
        hovermode='closest'
    )
    
    _add_control_menus(fig, df, treatments)
    
    # Glyphs legend 
    legend_text = (
        f'<span style="color:{CONFIG["GLYPH_COLORS"][1]}">&#9679;</span>=1 traitement | '
        f'<span style="color:{CONFIG["GLYPH_COLORS"][2]}">&#9670;</span>=2 traitements | '
        f'<span style="color:{CONFIG["GLYPH_COLORS"][3]}">&#9632;</span>=3 traitements | '
        f'<span style="color:{CONFIG["GLYPH_COLORS"][4]}">★</span>=4 traitements'
    )
    
    fig.add_annotation(
        text=legend_text,
        xref="paper", yref="paper",
        x=0.5,
        y=-0.025,
        showarrow=False,
        font=dict(size=15),
        bgcolor="white",
        yanchor="top"
    )
    fig.update_xaxes(title_text="", showgrid=False, showticklabels=False, zeroline=False, row=len(treatments) + 1)
    fig.update_yaxes(title_text="", showgrid=False, showticklabels=False, zeroline=False, row=len(treatments) + 1)
    
    fig.update_traces(
        row=len(treatments) + 1,
        col=1,
        patch={"subplot": "xy"},
        selector=dict(type="scatter")
    )
    fig.update_layout(
    {f"xaxis{len(treatments) + 1}": dict(showgrid=False, zeroline=False)},
    {f"yaxis{len(treatments) + 1}": dict(showgrid=False, zeroline=False)},
    paper_bgcolor='white',
    plot_bgcolor='white'
    )

def _add_control_menus(fig: go.Figure, df: pd.DataFrame, treatments: List[str]) -> None:
    """Adds interactive control menus."""
    fig.update_layout(updatemenus=[_create_heatmap_menu(df, treatments)])

def _create_heatmap_menu(df: pd.DataFrame, treatments: List[str]) -> dict:
    """Creates heatmap type selection menu."""
    return dict(
        buttons=create_heatmap_buttons(df, treatments),
        direction="down",
        showactive=True,
        x=1.1, y=1.05
    )

def update_heatmaps(fig: go.Figure, df: pd.DataFrame, treatments: List[str], mode: str = 'new'):
    """
    Met à jour les visualisations heatmap en fonction du mode sélectionné.
    
    Args:
        fig: Figure Plotly à mettre à jour
        df: DataFrame contenant les données patient
        treatments: Liste des traitements
        mode: 'new' ou 'active'
    """
    for i, trt in enumerate(treatments, 1):
        heatmap_data = get_heatmap_data(df, trt, mode)
        
        fig.update_traces(
            z=heatmap_data['z'],
            x=heatmap_data['x'],
            y=heatmap_data['y'],
            selector=dict(type='heatmap'),
            row=i, col=1
        )
        
        if mode == 'new':
            hovertemplate = "Début: sem. %{y}<br>Fin: sem. %{x}<br>Nombre de patients: %{z}<extra></extra>"
        else:
            hovertemplate = "Semaine: %{y}<br>Nombre de patients actifs: %{z}<extra></extra>"
            
        fig.update_traces(
            hovertemplate=hovertemplate,
            selector=dict(type='heatmap'),
            row=i, col=1
        )

# Application Controllers
def update_all_content(sample_size) -> tuple:
    """
    Updates visualization content based on patient sample data.
    
    Args:
        sample_size: Number of patients to sample. If not int, uses full dataset.
    Returns:
        tuple: (plotly figure, data records dict, column definitions)
    
    Raises:
        Exception: If processing fails
    """
    global df
    try:
        # Create and populate visualization
        fig = create_base_figure(df, CONFIG['TREATMENTS'])
        
        # Select patient sample (for glyphs and tale)
        sample_df = _get_patient_sample(sample_size)
        
        # Add glyphs and table
        _add_patient_traces(fig, sample_df)
        update_layout(fig, df, CONFIG['TREATMENTS']) 
        # Prepare table data
        columns = [{"name": col, "id": col} for col in sample_df.columns]
        return fig, sample_df.to_dict('records'), columns
    except Exception as e:
        print(f"Error updating content: {e}")
        raise

def _get_patient_sample(sample_size: int) -> pd.DataFrame:
    """Obtient un échantillon de patients en fonction de la taille spécifiée."""
    available_patients = sorted(df['Patient_ID'].unique())
    
    # Si sample_size est égal ou supérieur au nombre total de patients, retourner "tous"
    if sample_size >= len(available_patients):
        return df
    
    # Sinon sélectionner un échantillon aléatoire
    np.random.seed(42)  
    sample = np.random.choice(
        available_patients,
        size=sample_size,
        replace=False
    )
    return df[df['Patient_ID'].isin(sample)]

def _add_patient_traces(fig, sample_df: pd.DataFrame) -> None:
    patient_ids = sorted(sample_df['Patient_ID'].unique())
    weeks_range = _get_week_range(sample_df)
    max_week = max(weeks_range)
    
    # Tracer la grille de fond
    for w in weeks_range:
        fig.add_shape(
            type="line", x0=w, y0=0, x1=w, y1=len(patient_ids),
            line=dict(color="lightgray", width=0.5), row=len(CONFIG['TREATMENTS']) + 2, col=1
        )
    
    # Traces des patients
    for i, pid in enumerate(patient_ids):
        patient_df = sample_df[sample_df['Patient_ID'] == pid]
        
        # Ligne de base
        fig.add_trace(
            go.Scatter(x=weeks_range, y=[i] * len(weeks_range), mode='lines',
                    line=dict(color='gray', width=1, dash='dot'), showlegend=False),
            row=len(CONFIG['TREATMENTS']) + 2, col=1
        )
        
        # Pour chaque semaine, calculer les traitements actifs
        for week in weeks_range:
            active_treatments = patient_df[
                (patient_df['Start_Week'] <= week) &
                (patient_df['End_Week'] >= week)
            ]
            
            n_active_trts = len(active_treatments)
            if n_active_trts > 0:
                trt_names = ', '.join(active_treatments['Treatment'])
                fig.add_trace(
                    go.Scatter(
                        x=[week],
                        y=[i],
                        mode='markers',
                        marker=dict(
                            symbol=CONFIG['SYMBOLS'].get(n_active_trts),
                            size=CONFIG['GLYPH_BASE_SIZE'] + CONFIG['GLYPH_SIZE_INCREMENT'] * n_active_trts,
                            color=CONFIG['GLYPH_COLORS'].get(n_active_trts),
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=f"Patient {pid}<br>Semaine {week}<br>Traitements: {trt_names}<extra></extra>",
                        showlegend=False
                    ),
                    row=len(CONFIG['TREATMENTS']) + 2, col=1
                )

def validate_config():  # Configuration validation before app startup
    required_keys = ['N_PATIENTS', 'MAX_WEEK', 'TREATMENTS', 'COLORSCALE']
    if not all(key in CONFIG for key in required_keys):
        raise ValueError(f"Missing required config keys")

def init_application():
    global df
    df = generate_data(
        CONFIG['N_PATIENTS'], 
        CONFIG['MAX_WEEK'],
        CONFIG['TREATMENTS']
    )
    return df

if __name__ == '__main__':
    validate_config()
    init_application()  # Initialise la variable globale df
    app.run_server(debug=True)
