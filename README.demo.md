# Démonstrateur web TCA

Application Streamlit de démonstration pour le clustering de séquences unidimensionnelles et l’analyse multidimensionnelle SWoTTeD.

## Prérequis

- Docker avec le plugin Compose (pour le lancement conteneurisé), ou Python 3.11 pour l’installation locale.
- Les fichiers de démonstration livrés dans `data/`. Les fichiers importés par l’utilisateur restent traités localement par l’application.

## Lancement avec Docker

Depuis la racine du dépôt :

```bash
docker compose -f docker/docker-compose.yml up --build
```

Ouvrez [http://localhost:8501](http://localhost:8501). Pour arrêter le service, utilisez `Ctrl+C`, puis `docker compose -f docker/docker-compose.yml down`.

Le conteneur utilise un utilisateur non-root. Les exemples restent sous `data/` et le volume `external-data` est monté en lecture seule. Les fichiers CSV/XLSX présents dans ce dossier apparaissent comme choix « Externe » dans la barre latérale ; définissez `TCA_DATA_DIR` pour monter un autre dossier.

## Lancement local

Depuis la racine du dépôt, dans un environnement virtuel Python 3.11 :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m pip install setuptools wheel
python -m pip install --no-build-isolation --no-deps .
streamlit run app/streamlit_app.py
```

L’installation inclut PyTorch CPU pour SWoTTeD ; elle est donc plus volumineuse que la seule page unidimensionnelle.

## Pages

- **Unidimensionnel** : charge un CSV/XLSX large ou long, configure les colonnes, choisit Hamming, Optimal Matching ou Levenshtein, et exécute CAH, k-medoids ou k-means sur les fréquences. Les résultats présentent les affectations, des heatmaps par cluster, les pourcentages de statuts et, pour la CAH, un dendrogramme.
- **Multidimensionnel** : charge des observations individu/temps/événement et exécute une décomposition SWoTTeD légère, avec visualisation des intensités de phénotypes.

Les paramètres globaux (jeu d’exemple et limite d’individus) sont disponibles dans la barre latérale. Un import sur une page remplace le jeu d’exemple pour cette analyse.

## Limites et conseils de performance

- Les matrices de distances sont quadratiques ; la page unidimensionnelle est plafonnée à 300 individus, avec une limite globale réglable plus basse par défaut.
- Les doublons individu/temps au format long sont résolus en conservant la première observation, avec avertissement. Les valeurs manquantes de statut sont représentées par un état `(manquant)`.
- SWoTTeD s’exécute sur CPU et est plafonné à 100 individus, 60 périodes, 50 événements et 20 epochs. Un jeu plus petit et un rang/fenêtre réduits accélèrent le calcul.
- Le cache de session évite de recalculer les résultats pour les mêmes données et paramètres. Une modification de l’entrée ou des paramètres déclenche un nouveau calcul.
- Les métriques et visualisations restent exploratoires ; les séquences comportant le caractère `-` dans leurs statuts ne sont pas prises en charge par le codage TCA actuel.
- Le calcul multidimensionnel dépend du paquet SWoTTeD alpha et de PyTorch ; la compatibilité de modèles/données particuliers peut varier.

## Vérification rapide

```bash
python -m unittest discover -s tests
```

Pour vérifier l’application, lancez Docker puis exécutez un exemple CAH sur `unidimensional_data.csv`. L’application ne nécessite pas de compte ni de service externe.
