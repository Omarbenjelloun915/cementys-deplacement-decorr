# Décorrélation des déplacements – Projet Cementys

## Contenu

- `HSTPipeline.py` : script Python autonome. Il lit un fichier de données capteur (`dataset.dat`), applique le modèle HST (Hydrostatique – Saison – Temps), et enregistre les déplacements prédits et les résidus dans un CSV.
- `analyse_technique.ipynb` : notebook exploratoire contenant l’analyse EDA, les visualisations, et les tests de plusieurs modèles (régression, HST, MLP, XGBoost). Sert à analyser les résultats, comparer les performances et ajuster les paramètres.

## Objectif

Décorréler l’impact de la température et de l’ensoleillement sur les mesures de déplacement pour isoler la composante structurelle (vieillissement).

## Méthodes utilisées

- **Régression linéaire simple** : baseline interprétable.
- **HST (Hydrostatique–Saison–Temps)** : modélisation classique avec termes polynomiaux, sinus/cosinus pour la saisonnalité, et tendance lente. Bon compromis précision / interprétabilité.
- **MLP (réseau de neurones)** : capture les non-linéarités complexes. Meilleure performance brute mais modèle boîte noire.
- **XGBoost** : efficace, peu de tuning, alternative performante aux réseaux de neurones.

## Analyse & Résultats

L’analyse exploratoire montre une forte influence de la température et une saisonnalité horaire marquée.  
Les modèles non linéaires (MLP, XGBoost) atteignent un **R² > 0.91** en test avec un RMSE réduit.  
Le modèle HST, plus interprétable, offre des performances très proches tout en permettant une lecture directe des effets environnementaux.

**Choix retenu** : le modèle **HST** est privilégié pour sa robustesse, sa traçabilité et sa capacité à bien capter les composantes environnementales et temporelles, tout en permettant une analyse résiduelle fine du vieillissement.

## Automatisation

Le script `HSTPipeline.py` est structuré en classe (`HSTPipeline`), typé, et réutilisable. Il peut être intégré dans une pipeline d’analyse automatique. Le notebook permet d’explorer plus finement les effets ou de traiter d’autres capteurs.

## Exemple de lancement

```bash
python HSTPipeline.py dataset.dat
