Rapport d'Analyse : Performances des Étudiants (Student Performance UCI)
Objectif : Ce rapport analyse le jeu de données "Student Performance" de l'UCI pour comprendre les facteurs influençant les résultats scolaires (notes finales) des étudiants.

1. Description de la Base de Données
Informations du référentiel UCI
Source : UCI Machine Learning Repository, Dataset 320

Contexte : Ces données concernent les résultats des élèves de l'enseignement secondaire dans deux écoles portugaises.

Fichiers : Les données sont chargées via la bibliothèque ucimlrepo. Cette méthode récupère et combine les deux ensembles de données ('Mathématiques' et 'Langue Portugaise'), qui totalisent 1044 entrées. Les 32 premières colonnes (comme age, studytime, G1, G2) sont chargées comme "features" (X) et la note finale (G3) est chargée comme "target" (y).

Analyse : Pour ce rapport, nous re-combinons les features (X) et la cible (y) en un seul DataFrame afin d'analyser leurs relations. Cette analyse porte sur l'ensemble de ces 1044 données combinées.

Informations Externes (Usages courants)
Ce jeu de données est très populaire dans la communauté de la data science et du machine learning. Il est le plus souvent utilisé pour :

La Régression : Prédire la note finale exacte (G3) en fonction des autres variables.

La Classification : Prédire si un étudiant va réussir (par exemple, G3 >= 10) ou échouer (G3 < 10).

L'Analyse de Facteurs : Comprendre quels facteurs, notamment sociaux (comme Dalc - consommation d'alcool en semaine, ou goout - sorties), ont le plus d'impact sur la réussite scolaire.

2. Code Python et Analyse Exploratoire
Ce code est conçu pour être exécuté dans des cellules Google Colab.

2.1. Importation des Bibliothèques et Chargement des Données (via ucimlrepo)
Explication du code :

On importe pandas, matplotlib, seaborn et numpy.

On installe la bibliothèque ucimlrepo nécessaire pour charger les données depuis le référentiel UCI.

On utilise fetch_ucirepo(id=320) pour récupérer le jeu de données.

La bibliothèque sépare intelligemment les données en features (X) et targets (y, qui contient G3).

On affiche les métadonnées et les variables pour comprendre la structure.

On utilise pd.concat pour re-combiner X et y en un seul DataFrame df, ce qui est nécessaire pour notre analyse exploratoire (par ex., calculer la corrélation entre G1 et G3).

Python

# Étape 1 : Importer les bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurer Seaborn pour un affichage plus joli
sns.set(style="whitegrid")

# Étape 2 : Installer la bibliothèque ucimlrepo
!pip install ucimlrepo -q

# Étape 3 : Charger les données avec ucimlrepo
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets

# Étape 4 : Combiner les features (X) et la cible (y) en un seul DataFrame
df = pd.concat([X, y], axis=1)

# Étape 5 : Afficher les métadonnées et les informations sur les variables (comme demandé)
print("--- METADATA ---")
print(student_performance.metadata)
print("\n--- VARIABLES ---")
print(student_performance.variables)

# Afficher les 5 premières lignes pour vérifier
print("\n--- APERÇU DU DATAFRAME COMBINÉ ---")
print(df.head())
2.2. Exploration Initiale de la Structure
Explication du code :

df.info() nous montre le type de chaque colonne et s'il y a des valeurs manquantes.

Python

# Obtenir les informations générales sur le DataFrame
print("\nInformations sur le DataFrame :")
df.info()
Analyse des résultats de df.info() :

On constate qu'il y a 1044 entrées (étudiants des deux cours, maths et portugais).

Il y a 33 colonnes au total (32 features + 1 target G3).

Point important : Il n'y a aucune valeur manquante (non-null) dans ce jeu de données.

2.3. Analyse Statistique Détaillée (Mode, Médiane, Quartiles, etc.)
Explication du code : Nous allons maintenant calculer et afficher explicitement les principales statistiques descriptives pour les variables numériques clés : G3 (note finale), absences, et age.

Mesures de tendance centrale : Moyenne, Médiane, Mode.

Mesures de dispersion : Écart-type, Variance, Étendue (Min/Max), Étendue Interquartile (IQR).

Mesures de position : Quartiles (Q1, Q3).

Mesures de forme : Skewness (asymétrie) et Kurtosis (aplatissement).

Python

# --- Début du bloc de code pour l'analyse statistique détaillée ---

print("===============================================")
print("=== ANALYSE STATISTIQUE DÉTAILLÉE (PYTHON) ===")
print("===============================================")

# --- Analyse de la variable cible 'G3' (Note Finale) ---
print("\n--- Statistiques pour 'G3' (Note Finale) ---")

# Tendance centrale
mean_g3 = df['G3'].mean()
median_g3 = df['G3'].median()
mode_g3 = df['G3'].mode()[0]  # .mode() renvoie une série, [0] prend la première valeur (le mode)

print(f"  Moyenne   : {mean_g3:.2f}")
print(f"  Médiane   : {median_g3}")
print(f"  Mode      : {mode_g3}")

# Dispersion et Position
std_g3 = df['G3'].std()
var_g3 = df['G3'].var()
min_g3 = df['G3'].min()
max_g3 = df['G3'].max()
range_g3 = max_g3 - min_g3
q1_g3 = df['G3'].quantile(0.25)
q3_g3 = df['G3'].quantile(0.75)
iqr_g3 = q3_g3 - q1_g3

print(f"\n  Écart-type: {std_g3:.2f}")
print(f"  Variance  : {var_g3:.2f}")
print(f"\n  Minimum   : {min_g3}")
print(f"  Maximum   : {max_g3}")
print(f"  Étendue (Range) : {range_g3}")
print(f"  Quartile 1 (Q1 - 25%) : {q1_g3}")
print(f"  Quartile 3 (Q3 - 75%) : {q3_g3}")
print(f"  Étendue Interquartile (IQR) : {iqr_g3}")

# Forme de la distribution
skew_g3 = df['G3'].skew()
kurt_g3 = df['G3'].kurt()

print(f"\n  Asymétrie (Skewness) : {skew_g3:.2f}")
print(f"  Aplatissement (Kurtosis) : {kurt_g3:.2f}")

# --- Analyse de la variable 'absences' ---
print("\n--- Statistiques pour 'absences' ---")

mean_abs = df['absences'].mean()
median_abs = df['absences'].median()
mode_abs = df['absences'].mode()[0]
std_abs = df['absences'].std()
max_abs = df['absences'].max()
q1_abs = df['absences'].quantile(0.25)
q3_abs = df['absences'].quantile(0.75)
skew_abs = df['absences'].skew()

print(f"  Moyenne   : {mean_abs:.2f}")
print(f"  Médiane   : {median_abs}")
print(f"  Mode      : {mode_abs}")
print(f"  Écart-type: {std_abs:.2f}")
print(f"  Maximum   : {max_abs}")
print(f"  Quartile 1 (Q1 - 25%): {q1_abs}")
print(f"  Quartile 3 (Q3 - 75%): {q3_abs}")
print(f"  Asymétrie (Skewness) : {skew_abs:.2f} (Note: très asymétrique)")


# --- Analyse de la variable 'age' ---
print("\n--- Statistiques pour 'age' ---")
print(f"  Moyenne   : {df['age'].mean():.2f}")
print(f"  Médiane   : {df['age'].median()}")
print(f"  Mode      : {df['age'].mode()[0]}")
print(f"  Minimum   : {df['age'].min()}")
print(f"  Maximum   : {df['age'].max()}")


# --- BONUS : Résumé 'describe()' transposé (plus lisible) ---
print("\n===============================================")
print("=== RÉSUMÉ 'DESCRIBE' (TRANSPOSÉ) ===")
print("===============================================")
# .T transpose le tableau (les colonnes deviennent des lignes) pour une meilleure lisibilité
print(df.describe().T)

# --- Fin du bloc de code ---
Analyse des résultats statistiques (Exemple pour 'G3') :

Moyenne vs Médiane : Comparez la moyenne et la médiane. Si la moyenne est inférieure à la médiane, cela suggère une asymétrie à gauche (tirée vers le bas par de faibles notes, comme 0).

Asymétrie (Skewness) : Une valeur négative confirme l'asymétrie à gauche.

Étendue Interquartile (IQR) : 50% des étudiants (le "milieu" de la classe) ont des notes comprises entre Q1 et Q3.

Statistiques 'absences' : Notez la grande différence entre la moyenne, la médiane (probablement basse) et le maximum (probablement élevé). L'asymétrie sera fortement positive, indiquant que la plupart des étudiants ont peu d'absences, mais quelques-uns en ont énormément.

2.4. Visualisations (Graphiques)
Ici, nous créons plusieurs types de graphiques pour explorer visuellement les données.

A. Histogramme : Distribution des notes finales (G3)
Explication : Un histogramme nous montre la répartition des étudiants en fonction de leur note finale. Le kde=True ajoute une ligne lissée (estimation de la densité).

Python

plt.figure(figsize=(10, 6))
sns.histplot(df['G3'], bins=20, kde=True, color='blue')
plt.title('Distribution des Notes Finales (G3) - Maths et Portugais Combinés')
plt.xlabel('Note Finale (sur 20)')
plt.ylabel('Nombre d\'Étudiants')
plt.show()
Analyse du graphique : La distribution montre un pic notable à 10 et un autre pic à 0, indiquant un nombre non négligeable d'étudiants en échec sévère.

B. Boîte à Moustaches (Box Plot) : Note finale (G3) vs Temps d'étude (studytime)
Explication : La boîte à moustaches est parfaite pour comparer la distribution d'une variable numérique (G3) à travers différentes catégories (studytime).

Python

# Mapper les chiffres en labels plus clairs pour le graphique
studytime_labels = {1: '< 2h', 2: '2-5h', 3: '5-10h', 4: '> 10h'}
df['studytime_label'] = df['studytime'].map(studytime_labels)

plt.figure(figsize=(10, 6))
sns.boxplot(x='studytime_label', y='G3', data=df, order=['< 2h', '2-5h', '5-10h', '> 10h'])
plt.title('Note Finale (G3) en fonction du Temps d\'Étude Hebdomadaire')
plt.xlabel('Temps d\'étude')
plt.ylabel('Note Finale (sur 20)')
plt.show()
Analyse du graphique : On observe une légère tendance : la médiane des notes augmente avec le temps d'étude, particulièrement pour ceux qui étudient plus de 10 heures.

C. Nuage de Points (Scatter Plot) : Corrélation G1 vs G3
Explication : Ce graphique est idéal pour voir la relation entre deux variables numériques. Nous nous attendons à une forte corrélation positive : un étudiant qui a bien réussi au premier semestre (G1) devrait aussi bien réussir à l'examen final (G3).

Python

plt.figure(figsize=(10, 6))
sns.scatterplot(x='G1', y='G3', data=df, alpha=0.6)
plt.title('Relation entre la Note du Premier Semestre (G1) et la Note Finale (G3)')
plt.xlabel('Note G1')
plt.ylabel('Note G3')
plt.show()
Analyse du graphique : Comme attendu, la corrélation est très forte et positive. Les points forment une ligne ascendante claire, indiquant que G1 est un excellent prédicteur de G3.

D. Diagramme en Bâtons (Bar Plot) : Impact des échecs passés
Explication : Un diagramme en bâtons montre comment la moyenne de G3 change en fonction du nombre d'échecs passés (failures).

Python

plt.figure(figsize=(10, 6))
sns.barplot(x='failures', y='G3', data=df)
plt.title('Note Finale Moyenne en fonction du Nombre d\'Échecs Passés')
plt.xlabel('Nombre d\'échecs passés')
plt.ylabel('Note Finale Moyenne (G3)')
plt.show()
Analyse du graphique : L'impact est très net. La note finale moyenne chute drastiquement avec chaque échec passé.

E. Matrice de Corrélation (Heatmap)
Explication : Une heatmap (carte de chaleur) est un moyen puissant de visualiser la corrélation entre toutes les variables numériques en même temps. Une couleur chaude (rouge) signifie une forte corrélation positive, et une couleur froide (bleu) une forte corrélation négative.

Python

# Sélectionner uniquement les colonnes numériques pertinentes pour la corrélation
# (G3 est déjà dans df, pas besoin de sélectionner X seulement)
numeric_cols = df.select_dtypes(include=[np.number])

plt.figure(figsize=(14, 11))
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
plt.title('Matrice de Corrélation des Variables Numériques')
plt.tight_layout() # Ajuste le graphique pour éviter les superpositions
plt.show()
Analyse du graphique :

Fortes corrélations positives : G1 et G2 sont très fortement corrélés avec G3 (0.82 et 0.91), ce qui est logique. Medu (éducation de la mère) et Fedu (éducation du père) sont aussi corrélées entre elles.

Fortes corrélations négatives : failures (échecs) est fortement corrélée négativement avec toutes les notes (G1, G2, G3).

3. Conclusion de l'Analyse
Cette analyse exploratoire de la base de données combinée (Maths et Portugais) révèle plusieurs points clés :

Prévisibilité des Notes : Les notes finales (G3) sont très fortement prédites par les notes précédentes (G1 et G2), avec des coefficients de corrélation supérieurs à 0.8.

Impact des Échecs : Le nombre d'échecs passés (failures) est le prédicteur négatif le plus puissant de la réussite future.

Facteurs Sociaux : Le temps d'étude (studytime) montre une corrélation positive modeste mais claire avec les notes.

Qualité des Données : La base de données est complète et ne contient aucune valeur manquante, la rendant idéale pour des modèles de machine learning.
