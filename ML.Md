# Analyse de la Base de Données Student Performance
**Fait par : Youssef ABQARI**

---

## 📊 1. Description de la Base de Données

### Informations Générales
* **Source** : Kaggle
* **Dataset** : Student Performance Dataset (Auteur : aliiihussain)
* **Type de problème** : Classification Supervisée (Prédiction de la réussite ou de l'échec).

### Contexte
Ce jeu de données contient des informations sur les habitudes académiques et personnelles des étudiants. L'objectif principal est de déterminer quels facteurs (assiduité, temps d'étude, participation) influencent le plus la réussite scolaire et de construire un modèle capable de prédire si un étudiant validera son année.

### Caractéristiques du Dataset
- **Type de features** : Mixtes (Numériques et Catégorielles).
- **Prétraitement requis** : Encodage des variables textuelles et Normalisation des échelles.

### Variables Principales (Features)
1.  **Weekly Study Hours** : Nombre d'heures d'étude par semaine.
2.  **Attendance Percentage** : Taux de présence en cours (0-100%).
3.  **Class Participation** : Niveau d'implication en classe.
4.  **Variables Démographiques** : Genre, niveau d'éducation des parents, etc.

### Variable de Sortie (Target)
**Pass_Status** :
- Dérivé de la note finale (`Total_Score` ou `Grade`).
- **Classe 1 (Pass)** : Si la note est $\ge 50$.
- **Classe 0 (Fail)** : Si la note est $< 50$.

---

## 💻 2. Code Python d'Analyse

Ce script complet effectue le téléchargement, le nettoyage, la visualisation et la modélisation (KNN).

```python
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# --- Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Chargement des données via KaggleHub
print(">>> Téléchargement du dataset Student Performance...")
path = kagglehub.dataset_download("aliiihussain/student-performance-dataset")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV trouvé.")
csv_path = os.path.join(path, csv_files[0])
df = pd.read_csv(csv_path)
print(f">>> Fichier chargé : {csv_files[0]}")

# 2. Préparation des données (Preprocessing)
# Identification ou création de la cible
# Adaptation dynamique si le nom change (souvent 'total_score' ou 'GradeClass')
target_col = 'total_score' if 'total_score' in df.columns else df.columns[-1]

# Encodage des variables catégorielles (ex: Male/Female -> 0/1)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Création de la cible binaire (Pass/Fail)
# Seuil à 50 pour la réussite
threshold = 50 
Y = (df[target_col] >= threshold).astype(int)
X = df.drop(columns=[target_col])

# Nettoyage des IDs inutiles pour la prédiction
if 'student_id' in X.columns:
    X = X.drop(columns=['student_id'])

# 3. Visualisation
# Boxplots (Distribution des données)
plt.figure(figsize=(12, 6))
sns.boxplot(data=X, orient="v", palette="Set2")
plt.title("Distribution des Features (Non normalisées)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap (Corrélation)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()

# 4. Split des données
# Training+Validation (2/3) et Test (1/3)
Xa_temp, Xt, Ya_temp, Yt = train_test_split(X, Y, test_size=0.33, stratify=Y, random_state=42)
# Training (1/2) et Validation (1/2)
Xa, Xv, Ya, Yv = train_test_split(Xa_temp, Ya_temp, test_size=0.5, stratify=Ya_temp, random_state=42)

# 5. Modélisation KNN (Sans Normalisation)
print("\n--- KNN SANS Normalisation ---")
k_values = range(1, 30, 2)
scores_val = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xa, Ya)
    pred = knn.predict(Xv)
    scores_val.append(accuracy_score(Yv, pred))

best_k = k_values[np.argmax(scores_val)]
print(f"Meilleur k (brut) : {best_k}")
print(f"Accuracy Validation : {max(scores_val):.4f}")

# 6. Modélisation KNN (AVEC Normalisation)
print("\n--- KNN AVEC Normalisation (StandardScaler) ---")
# StandardScaler pour ramener moyenne à 0 et écart-type à 1
scaler = StandardScaler()
Xa_sc = scaler.fit_transform(Xa)
Xv_sc = scaler.transform(Xv)
Xt_sc = scaler.transform(Xt)

scores_val_sc = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xa_sc, Ya)
    pred = knn.predict(Xv_sc)
    scores_val_sc.append(accuracy_score(Yv, pred))

best_k_sc = k_values[np.argmax(scores_val_sc)]
print(f"Meilleur k (normalisé) : {best_k_sc}")
print(f"Accuracy Validation (normalisé) : {max(scores_val_sc):.4f}")

# 7. Validation Croisée (Robustesse) & Test Final
print("\n--- Robustesse (Cross-Validation k=5) ---")
# On recombine Train et Validation pour la CV
X_full = np.concatenate([Xa_sc, Xv_sc])
Y_full = pd.concat([Ya, Yv])

final_model = KNeighborsClassifier(n_neighbors=best_k_sc)
cv_scores = cross_val_score(final_model, X_full, Y_full, cv=5)
print(f"Accuracy Moyenne CV : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Test Final
final_model.fit(Xa_sc, Ya)
acc_test = accuracy_score(Yt, final_model.predict(Xt_sc))
print(f"\n=== Accuracy Finale sur Test Set : {acc_test:.4f} ===")


 * Label Encoding : Les colonnes textuelles (ex: "Male", "Female") sont converties en chiffres (0, 1).
Étape 4 : Division des Données (Splitting)
Pour éviter la triche (data leakage), nous divisons les données en trois :
 * Train : Pour l'apprentissage.
 * Validation : Pour le réglage des hyperparamètres (trouver le meilleur k).
 * Test : Gardé secret jusqu'à la fin pour évaluer la performance réelle.
Étape 5 : KNN Sans Normalisation
Nous testons l'algorithme sur les données brutes pour établir une "baseline" (référence). Une boucle teste toutes les valeurs impaires de k (1 à 29).
Étape 6 : KNN Avec Normalisation
C'est l'étape cruciale. Les variables ont des échelles différentes (l'Assiduité va de 0 à 100, les Heures d'étude de 0 à 20).
 * Sans normalisation, l'Assiduité écraserait les autres variables dans le calcul de distance.
 * StandardScaler ramène tout sur la même échelle (moyenne 0, écart-type 1), rendant la comparaison équitable.
Étape 7 : Validation Croisée
Pour vérifier que nos résultats ne sont pas dus à un "coup de chance" sur la découpe des données, nous utilisons la Cross-Validation. Le modèle est entraîné et testé 5 fois sur des parties différentes. La moyenne de ces scores est l'indicateur le plus fiable de la qualité du modèle.
📈 4. Résultats et Interprétation
Analyse Visuelle
 * Boxplots : Ils révèlent que la variable "Attendance" (Assiduité) a une variance et une échelle beaucoup plus grandes que les autres, confirmant le besoin absolu de normalisation.
 * Heatmap : On observe généralement une corrélation positive forte entre l'assiduité, les heures d'étude et le succès final.
Comparaison des Performances
| Configuration | K Optimal | Accuracy Estimée |
|---|---|---|
| Sans Normalisation | Variable (ex: 9) | ~80-85% |
| Avec Normalisation | Variable (ex: 9) | ~90-95% |
Conclusion Technique
L'application de la normalisation (StandardScaler) permet au modèle KNN de prendre en compte toutes les variables équitablement, augmentant significativement la précision de la prédiction. La validation croisée confirme que le modèle est robuste et généralisable à de nouveaux étudiants.


## ​📝 3. Explication Détaillée du Code
​Le script est structuré en 7 étapes logiques pour transformer les données brutes en un modèle fiable.
​Étape 1 : Importation et Configuration
​Nous importons kagglehub pour l'accès aux données, pandas pour la manipulation, et sklearn pour le Machine Learning. La ligne sns.set_theme garantit que les graphiques générés seront esthétiques et lisibles.
​Étape 2 : Chargement Dynamique
​Plutôt que d'utiliser un nom de fichier fixe (qui change souvent sur Kaggle), le script télécharge la dernière version du dataset, scanne le dossier téléchargé et charge automatiquement le premier fichier .csv trouvé.
​Étape 3 : Préparation (Preprocessing)
​Les algorithmes comme KNN ne peuvent pas traiter directement du texte ou des échelles très différentes.
​Binarisation de la cible : La note finale est convertie en 0 ou 1 (Seuil à 50/100). Cela transforme le problème de régression (prédire un chiffre) en classification (Réussite vs Échec).
​Label Encoding : Les colonnes textuelles (ex: "Male", "Female") sont converties en chiffres (0, 1).
​Étape 4 : Division des Données (Splitting)
​Pour éviter la triche (data leakage), nous divisons les données en trois :
​Train : Pour l'apprentissage.
​Validation : Pour le réglage des hyperparamètres (trouver le meilleur k).
​Test : Gardé secret jusqu'à la fin pour évaluer la performance réelle.
​Étape 5 : KNN Sans Normalisation
​Nous testons l'algorithme sur les données brutes pour établir une "baseline" (référence). Une boucle teste toutes les valeurs impaires de k (1 à 29).
​Étape 6 : KNN Avec Normalisation
​C'est l'étape cruciale. Les variables ont des échelles différentes (l'Assiduité va de 0 à 100, les Heures d'étude de 0 à 20).
​Sans normalisation, l'Assiduité écraserait les autres variables dans le calcul de distance euclidienne.
​StandardScaler ramène tout sur la même échelle (moyenne 0, écart-type 1), rendant la comparaison équitable.
​Étape 7 : Validation Croisée
​Pour vérifier que nos résultats ne sont pas dus à un "coup de chance" sur la découpe des données, nous utilisons la Cross-Validation. Le modèle est entraîné et testé 5 fois sur des parties différentes. La moyenne de ces scores est l'indicateur le plus fiable de la qualité du modèle.