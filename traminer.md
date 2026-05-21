https://traminer.unige.ch/

- TraMineR traite des séquences d'états et repose sur 4 grands piliers 
    - Définition de l'alphabet (tous les états possibles)
    - Le calcul des distances / dissimilarités comme tca
    - Les statistiques longitudinales qui permettent de résumer une trajectoire :
        - L'entropie de Shannon : pour mesurer la diversité des états dans le temps $H = -\sum p_i log(p_i)$ où $p_i$ est la proportion de temps passé dans l'état $i$
        - La turbulence qui prend en compte l'entropie et les nombres de transitions
    - La visualisation : 
        - Index plot : chaque ligne est un inidividu, les couleurs représentent les états au fil du temps
        - Chronogrammes (state distribution plots) : montre la proportion d'individus dans chaque état à chaque instant $t$
        - Modal plots : affiche la trajectoire "modale" (l'etat le plus fréquent à chaque instant $t$)

- La préparation de données est faite à partir de fonctions utilitaires, on peut s'en inspirer pour convertir les données longues en données larges et gérer le padding

- Dans ce package, pour Optimal Matching, le calcul est très lourd (conseil IA : utiliser des bibliothèques existantes comme `textdistance` ou la fonction `pdist`de `scipy.spatial.distance`)
