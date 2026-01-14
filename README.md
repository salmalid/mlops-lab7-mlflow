# Lab 7 : Intégration MLflow pour le Suivi et la Gestion des Modèles

## Introduction

Ce Lab a pour objectif d'intégrer **MLflow** dans un pipeline MLOps pour :
-  Tracer les expérimentations (paramètres, métriques, artefacts)
-  Gérer le cycle de vie des modèles via le Model Registry
-  Automatiser la promotion et le rollback des modèles
-  Intégrer le modèle actif dans une API de prédiction

## Architecture du Projet

```
mlops-lab7-mlflow/
├── .dvc/                      # Configuration DVC
├── .github/workflows/         # CI/CD pipelines
├── data/
│   ├── raw.csv               # Dataset brut (suivi par DVC)
│   └── processed.csv         # Dataset traité (suivi par DVC)
├── mlflow/
│   ├── mlflow.db             # Base de données SQLite (tracking)
│   └── artifacts/            # Stockage des artefacts MLflow
├── models/
│   └── model.joblib          # Modèles exportés (suivi par DVC)
├── registry/
│   ├── metadata.json         # Métadonnées des modèles
│   └── current_model.txt     # Modèle actif
├── reports/
│   └── metrics.json          # Métriques d'évaluation
├── src/
│   ├── prepare_data.py       # Préparation des données
│   ├── train.py              # Entraînement avec MLflow
│   ├── evaluate.py           # Évaluation du modèle
│   ├── promote.py            # Promotion vers production
│   ├── rollback.py           # Rollback de version
│   └── api.py                # API FastAPI
├── k8s/                      # Configurations Kubernetes
├── dvc.yaml                  # Pipeline DVC
├── dvc.lock                  # État du pipeline DVC
└── requirements.txt          # Dépendances Python
```

## Étape 1 : Initialisation de l'environnement et installation de MLflow
Initialiser un environnement Python isolé et installer MLflow.

<img width="948" height="353" alt="etape 1" src="https://github.com/user-attachments/assets/563cb909-254d-45c5-90ca-dbaa9d3800a2" />


## Étape 2 : Création explicite de l'espace de stockage MLflow
Créer une structure dédiée au stockage des métadonnées et artefacts MLflow.

<img width="958" height="800" alt="etape 2" src="https://github.com/user-attachments/assets/4dde0383-e8ef-434e-a286-2d0e9ef6b3ba" />


## Étape 3 : Configuration du client MLflow
Configurer l'URI du tracking server pour centraliser les expérimentations.

Tous les scripts MLflow enverront désormais leurs runs vers le serveur central au lieu de créer des dossiers `mlruns/` locaux.

<img width="948" height="191" alt="etape 3" src="https://github.com/user-attachments/assets/86045475-37b1-457f-921e-6541b08235e0" />


## Étape 4 : Démarrage du serveur MLflow (tracking server)
Démarrer un serveur MLflow local avec SQLite comme backend de métadonnées.

Le tracking server devient la **source centrale de vérité** pour :
- Les expérimentations (runs)
- Les modèles enregistrés (Model Registry)
- Les métriques et paramètres

<img width="937" height="291" alt="etape 4" src="https://github.com/user-attachments/assets/119c1017-bd8e-4aa8-89df-f60e11696a2f" />

<img width="948" height="792" alt="etape 4-2" src="https://github.com/user-attachments/assets/c84384a3-edc9-4833-922b-d97f031eb1a0" />


## Étape 5 : Instrumentation réelle de train.py
Modifier `src/train.py` pour enregistrer automatiquement dans MLflow :
1. Les paramètres et métriques
2. Le fichier modèle exporté
3. Une version du modèle dans le Model Registry

**Interface MLflow :**
- Un run apparaît dans l'expérience `mlops-lab-01`
- Paramètres : `version`, `seed`, `gate_f1`
- Métriques : F1, accuracy..
- Artefacts : fichier `.joblib` dans `exported_models/`
- Modèle `churn_model` version 1 dans le Model Registry

<img width="937" height="863" alt="etape 5" src="https://github.com/user-attachments/assets/b353d92b-2309-453e-ab0b-81a44ca5d7c5" />

<img width="898" height="952" alt="etape 5 -2" src="https://github.com/user-attachments/assets/ad1e94dd-b404-4f62-aefc-b1cfd89a895e" />

## Étape 6 : Observation du registry MLflow
Observer le modèle enregistré dans l'interface MLflow.

Le Model Registry MLflow remplit le rôle conceptuel de **registre de modèles** :
- **Versioning** : chaque entraînement crée une version incrémentale
- **Traçabilité** : lien avec le run d'expérimentation complet
- **Lifecycle** : états (Staging, Production, Archived)
- **Alias** : pointeurs nommés vers des versions spécifiques
  
<img width="951" height="812" alt="etape 6" src="https://github.com/user-attachments/assets/5ad8ccd6-7146-42cc-8bce-0e52d9553ed4" />


## Étape 7 : Promotion d'un modèle (activation)
Créer un script pour promouvoir automatiquement la dernière version en production.

### Explication du concept d'alias

Un **alias MLflow** est un pointeur nommé vers une version spécifique :
- `production` : version déployée en production
- `staging` : version en cours de test
- `champion` : meilleure version connue

**Avantages :**
- Changement de version sans modification de code
- Traçabilité de ce qui est déployé
- Rollback simplifié

<img width="940" height="291" alt="etape 7 - 1" src="https://github.com/user-attachments/assets/395b0643-668b-465b-bf29-206455014ebc" />

<img width="938" height="155" alt="etape 7" src="https://github.com/user-attachments/assets/e1945a00-80c1-4b65-a6c1-e4c12bbc7589" />


## Étape 8 : Rollback via MLflow Model Registry
Implémenter un mécanisme de rollback pour revenir à une version antérieure du modèle.

retrain : 

<img width="957" height="497" alt="etape 8 - retrain" src="https://github.com/user-attachments/assets/24eec9c5-d818-49f6-b046-58abc56d1ad7" />

rollback : 

<img width="946" height="375" alt="etape 8 rollback" src="https://github.com/user-attachments/assets/fd5fd371-7083-48f9-9d3d-04f42e30bcdc" />

<img width="945" height="202" alt="etape 8" src="https://github.com/user-attachments/assets/7b9e0087-da7f-4b7b-bfda-bb7d20fad10c" />


## Étape 9 : API – chargement du modèle actif
Adapter l'API FastAPI pour charger automatiquement le modèle actif depuis MLflow.

### Résultat attendu

-  L'API charge toujours la version pointée par `production`
-  Pas besoin de modifier le code pour changer de modèle
-  Traçabilité complète : quelle version sert les prédictions

<img width="882" height="706" alt="etape 9" src="https://github.com/user-attachments/assets/e9858b5c-55a8-4119-93a6-b8ed2189edc1" />


Ce laboratoire a permis de :
1.  **Tracer** les expérimentations avec MLflow Tracking
2.  **Versionner** les modèles dans le Model Registry
3.  **Automatiser** la promotion et le rollback
4.  **Intégrer** le modèle actif dans une API de production

### Architecture MLOps complète

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   DVC       │────▶│   MLflow    │────▶│  FastAPI     │
│ (Données)   │     │  (Modèles)  │     │ (Prédictions)│
└─────────────┘     └─────────────┘     └──────────────┘
      │                    │                     │
      ▼                    ▼                     ▼
  Versioning         Registry             Production
```
<img width="952" height="837" alt="final - MLflow " src="https://github.com/user-attachments/assets/2114dcd2-a959-472f-ac15-5ef7433d0e1e" />


## Auteur

**Salma Lidame**  

AI Engineer - ENSA El Jadida

Cours : MLOps
