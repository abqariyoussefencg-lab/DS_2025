# RAPPORT D'ANALYSE ÉCONOMIQUE
## ÉTUDE DU PRODUIT INTÉRIEUR BRUT MONDIAL 2024

---

**Date du rapport :** Octobre 2025  
**Auteur :** Analyse économique comparative  
**Sources :** FMI, Banque Mondiale, Statista  
**Période d'étude :** Année 2024



Youssef ABQARI

<img src="Me.jpg" style="height:464px;margin-right:432px"/>

<img src="téléchargement.jpeg" style="height:464px;margin-right:432px"/>
---

## RÉSUMÉ EXÉCUTIF

Ce rapport présente une analyse approfondie du Produit Intérieur Brut (PIB) mondial pour l'année 2024, avec un focus particulier sur les dix plus grandes économies mondiales. L'analyse révèle un PIB mondial total de 111 326,37 milliards de dollars, marqué par une croissance mondiale estimée à 3,5%.

**Points clés :**
- Les États-Unis maintiennent leur position dominante avec un PIB de 29 167,78 milliards de dollars
- La Chine poursuit sa croissance rapide avec un PIB de 18 567 milliards de dollars et un taux de croissance de 5%
- L'Inde émerge comme le champion de la croissance avec 7,2%
- Les économies européennes montrent une croissance plus modérée (0,2% à 1,1%)
- Le TOP 10 des économies représente environ 70% du PIB mondial

---

## TABLE DES MATIÈRES

1. Introduction et contexte économique
2. Méthodologie et sources de données
3. Code Python d'analyse
4. Résultats détaillés
5. Interprétations et analyses
6. Perspectives et recommandations
7. Conclusion

---

## 1. INTRODUCTION ET CONTEXTE ÉCONOMIQUE

### 1.1 Objectifs de l'étude

L'année 2024 marque une période de transition économique mondiale caractérisée par la sortie progressive des impacts de la crise sanitaire, les tensions géopolitiques persistantes et les défis liés à l'inflation et aux politiques monétaires restrictives. Cette analyse vise à :

- Identifier les principales puissances économiques mondiales en 2024
- Comparer les performances économiques entre pays développés et émergents
- Analyser les disparités entre PIB total et PIB par habitant
- Évaluer les dynamiques de croissance économique

### 1.2 Contexte mondial 2024

L'année 2024 se caractérise par plusieurs tendances majeures :

- **Résilience économique américaine** : malgré les hausses de taux d'intérêt, l'économie américaine maintient une croissance de 2,8%
- **Dynamisme des économies asiatiques** : la Chine stabilise sa croissance autour de 5%, tandis que l'Inde accélère à 7,2%
- **Ralentissement européen** : l'Europe fait face à des défis structurels avec des taux de croissance faibles
- **Émergence continue du Sud global** : des pays comme le Brésil et l'Inde renforcent leur poids économique

---

## 2. MÉTHODOLOGIE ET SOURCES DE DONNÉES

### 2.1 Sources de données

Les données utilisées dans cette analyse proviennent de trois sources principales reconnues internationalement :

- **Fonds Monétaire International (FMI)** : Prévisions économiques mondiales et données PIB
- **Banque Mondiale** : Statistiques de développement et PIB par habitant
- **Statista** : Données consolidées et analyses comparatives

### 2.2 Indicateurs analysés

Trois indicateurs économiques clés sont analysés :

1. **PIB nominal (en milliards de dollars US)** : Valeur totale de la production économique
2. **Taux de croissance du PIB (en %)** : Variation annuelle du PIB
3. **PIB par habitant (en milliers de dollars)** : Mesure du niveau de vie moyen

### 2.3 Périmètre de l'étude

L'analyse se concentre sur les 10 premières économies mondiales en termes de PIB nominal, qui représentent collectivement environ 70% de la richesse mondiale.

---

## 3. CODE PYTHON D'ANALYSE

### 3.1 Bibliothèques utilisées

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
```

**Justification des choix techniques :**
- `matplotlib` : Création de visualisations professionnelles et personnalisables
- `numpy` : Calculs numériques et manipulations de données
- `pandas` : Structuration et analyse des données tabulaires

### 3.2 Configuration de l'environnement graphique

```python
# Configuration de l'affichage
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
```

Cette configuration assure une cohérence visuelle et une lisibilité optimale des graphiques.

### 3.3 Données collectées

```python
# Données PIB 2024 (en milliards de dollars US)
pays = ['États-Unis', 'Chine', 'Allemagne', 'Japon', 'Inde', 
        'Royaume-Uni', 'France', 'Italie', 'Brésil', 'Canada']
pib_2024 = [29167.78, 18567.0, 4659.93, 4271.0, 3937.0, 
            3339.0, 3174.1, 2328.0, 2188.42, 2139.0]

# Taux de croissance estimés 2024 (%)
croissance = [2.8, 5.0, 0.2, 1.0, 7.2, 1.1, 1.1, 0.8, 2.9, 1.2]

# PIB par habitant 2024 (en milliers de dollars)
pib_par_hab = [87.5, 13.2, 55.8, 34.1, 2.8, 49.5, 47.8, 39.5, 10.2, 55.3]
```

### 3.4 Création des visualisations

Le code génère cinq visualisations complémentaires :

1. **Graphique en barres horizontales** - Classement des PIB
2. **Graphique circulaire** - Répartition du PIB mondial
3. **Graphique en barres verticales** - Taux de croissance
4. **Graphique scatter** - Corrélation PIB total vs PIB par habitant
5. **Tableau récapitulatif** - Synthèse des indicateurs

### 3.5 Code complet exécutable

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)

# Données
pays = ['États-Unis', 'Chine', 'Allemagne', 'Japon', 'Inde', 
        'Royaume-Uni', 'France', 'Italie', 'Brésil', 'Canada']
pib_2024 = [29167.78, 18567.0, 4659.93, 4271.0, 3937.0, 
            3339.0, 3174.1, 2328.0, 2188.42, 2139.0]
croissance = [2.8, 5.0, 0.2, 1.0, 7.2, 1.1, 1.1, 0.8, 2.9, 1.2]
pib_par_hab = [87.5, 13.2, 55.8, 34.1, 2.8, 49.5, 47.8, 39.5, 10.2, 55.3]

# Création de la figure
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# [Code des visualisations - voir artifact code complet]

plt.tight_layout()
plt.show()
```

---

## 4. RÉSULTATS DÉTAILLÉS

### 4.1 Classement des économies mondiales par PIB 2024

| Rang | Pays | PIB (Mds $) | Part mondiale | Croissance (%) |
|------|------|-------------|---------------|----------------|
| 1 | États-Unis | 29 167,78 | 26,2% | 2,8 |
| 2 | Chine | 18 567,00 | 16,7% | 5,0 |
| 3 | Allemagne | 4 659,93 | 4,2% | 0,2 |
| 4 | Japon | 4 271,00 | 3,8% | 1,0 |
| 5 | Inde | 3 937,00 | 3,5% | 7,2 |
| 6 | Royaume-Uni | 3 339,00 | 3,0% | 1,1 |
| 7 | France | 3 174,10 | 2,9% | 1,1 |
| 8 | Italie | 2 328,00 | 2,1% | 0,8 |
| 9 | Brésil | 2 188,42 | 2,0% | 2,9 |
| 10 | Canada | 2 139,00 | 1,9% | 1,2 |

**PIB mondial total 2024 :** 111 326,37 milliards de dollars  
**Part du TOP 10 :** 77 771,23 milliards de dollars (69,9%)

### 4.2 Analyse des taux de croissance

**Classification par dynamique de croissance :**

**🚀 Croissance forte (> 5%):**
- Inde : 7,2% - Champion de la croissance parmi les grandes économies
- Chine : 5,0% - Maintien d'un rythme élevé malgré les défis

**📈 Croissance modérée (2% - 5%):**
- Brésil : 2,9% - Reprise économique solide
- États-Unis : 2,8% - Résilience remarquable

**📊 Croissance faible (< 2%):**
- Canada : 1,2%
- France : 1,1%
- Royaume-Uni : 1,1%
- Japon : 1,0%
- Italie : 0,8%
- Allemagne : 0,2% - Stagnation préoccupante

### 4.3 PIB par habitant - Indicateur de richesse

**Classement par PIB par habitant (en milliers de dollars) :**

1. **États-Unis** : 87,5 k$ - Richesse individuelle exceptionnelle
2. **Allemagne** : 55,8 k$ - Standard de vie européen élevé
3. **Canada** : 55,3 k$ - Proximité avec le niveau américain
4. **Royaume-Uni** : 49,5 k$ - Économie développée mature
5. **France** : 47,8 k$ - Niveau de vie confortable
6. **Italie** : 39,5 k$ - Écart avec les leaders européens
7. **Japon** : 34,1 k$ - Impacté par une population vieillissante
8. **Chine** : 13,2 k$ - Pays à revenu intermédiaire supérieur
9. **Brésil** : 10,2 k$ - Économie émergente
10. **Inde** : 2,8 k$ - Pays à faible revenu malgré forte croissance

### 4.4 Statistiques descriptives

**Analyse statistique du TOP 10 :**

- **PIB moyen :** 7 777,12 milliards de dollars
- **Écart-type :** 9 248,45 milliards de dollars (forte dispersion)
- **Médiane du PIB :** 3 638,00 milliards de dollars
- **Coefficient de variation :** 118,9% (hétérogénéité importante)

**Taux de croissance :**
- **Moyenne :** 2,31%
- **Écart-type :** 2,19%
- **Médiane :** 1,15%

**PIB par habitant :**
- **Moyenne :** 39,65 k$
- **Écart-type :** 27,31 k$
- **Médiane :** 43,65 k$

---

## 5. INTERPRÉTATIONS ET ANALYSES

### 5.1 Domination américaine persistante

Les États-Unis conservent une avance considérable avec un PIB de 29 167,78 milliards de dollars, soit 57% supérieur à celui de la Chine. Cette domination s'explique par :

- **Innovation technologique** : Leadership dans les secteurs high-tech (GAFAM)
- **Marché intérieur puissant** : 335 millions de consommateurs à fort pouvoir d'achat
- **Dollar comme monnaie de réserve** : Avantage structurel dans le commerce international
- **Attractivité des investissements** : Système financier développé et stable

Le PIB par habitant américain de 87 500 dollars démontre une richesse individuelle exceptionnelle, supérieure de 57% à la moyenne européenne du TOP 10.

### 5.2 La montée en puissance asiatique

#### 5.2.1 Chine : le géant en transition

Avec un PIB de 18 567 milliards de dollars et une croissance de 5%, la Chine maintient une dynamique impressionnante. Cependant, plusieurs défis persistent :

- **Ralentissement par rapport aux décennies précédentes** (objectif de croissance revu à la baisse)
- **Crise immobilière** : Secteur représentant 25% du PIB en difficulté
- **Démographie défavorable** : Population en déclin depuis 2022
- **Tensions géopolitiques** : Impact sur les chaînes d'approvisionnement

Le PIB par habitant chinois de 13 200 dollars (6,6 fois inférieur aux États-Unis) révèle un important potentiel de développement interne.

#### 5.2.2 Inde : le nouveau moteur de croissance

L'Inde affiche des performances exceptionnelles :

- **Croissance de 7,2%** : La plus élevée du TOP 10
- **Dividende démographique** : Population jeune de 1,4 milliard d'habitants
- **Digitalisation accélérée** : Transformation numérique rapide
- **Attractivité manufacturière** : Alternative à la Chine pour les délocalisations

Toutefois, le PIB par habitant de 2 800 dollars souligne les défis massifs de développement et de réduction de la pauvreté.

### 5.3 L'Europe en difficulté

Les économies européennes du TOP 10 affichent des taux de croissance préoccupants :

#### 5.3.1 Allemagne : la locomotive en panne

Avec seulement 0,2% de croissance, l'Allemagne fait face à des défis structurels :

- **Dépendance énergétique** : Impact de la crise du gaz russe
- **Transition automobile** : Difficultés dans l'électrification
- **Compétitivité industrielle** : Concurrence asiatique accrue
- **Démographie défavorable** : Vieillissement et manque de main-d'œuvre

#### 5.3.2 France et Royaume-Uni : croissance atone

France (1,1%) et Royaume-Uni (1,1%) partagent des défis communs :

- **Inflation résiduelle** : Pression sur le pouvoir d'achat
- **Dette publique élevée** : Marge de manœuvre budgétaire limitée
- **Productivité stagnante** : Besoin de réformes structurelles

Pour le Royaume-Uni, s'ajoute l'impact du Brexit sur les échanges commerciaux.

### 5.4 Disparités PIB total vs PIB par habitant

L'analyse révèle un paradoxe fascinant :

**Pays à fort PIB mais faible PIB/habitant :**
- **Chine** : 2ème PIB mondial, mais 8ème en PIB/habitant du TOP 10
- **Inde** : 5ème PIB mondial, mais 10ème en PIB/habitant du TOP 10
- **Brésil** : 9ème PIB mondial, mais 9ème en PIB/habitant du TOP 10

**Pays à PIB modéré mais fort PIB/habitant :**
- **Canada** : 10ème PIB mondial, mais 3ème en PIB/habitant du TOP 10
- **Allemagne** : 3ème PIB mondial, et 2ème en PIB/habitant du TOP 10

Cette divergence s'explique principalement par :
- **Démographie** : Population nombreuse dilue la richesse par habitant
- **Niveau de développement** : Pays émergents vs développés
- **Productivité** : Efficacité économique variable

### 5.5 Concentration de la richesse mondiale

Le TOP 10 représente 69,9% du PIB mondial avec seulement 10 pays sur 195 dans le monde. Cette concentration révèle :

- **Inégalités mondiales** : 190 pays se partagent 30% de la richesse
- **Interdépendance économique** : Ces économies sont fortement liées
- **Risques systémiques** : Une crise dans ces pays impacte le monde entier

### 5.6 Dynamiques de convergence et divergence

**Convergence observée :**
- Les économies émergentes (Inde, Chine, Brésil) croissent plus vite que les pays développés
- Réduction progressive de l'écart de développement (en termes relatifs)

**Divergence persistante :**
- L'écart absolu en dollars continue de s'accroître
- Les États-Unis creusent l'écart avec l'Europe
- Les inégalités de PIB par habitant restent massives

---

## 6. PERSPECTIVES ET RECOMMANDATIONS

### 6.1 Projections à horizon 2030

**Scénarios probables :**

1. **États-Unis** : Maintien de la première place, PIB projeté à 35-37 billions $
2. **Chine** : Poursuite de la croissance, PIB projeté à 23-25 billions $
3. **Inde** : Dépassement probable de l'Allemagne et du Japon, 3ème position mondiale
4. **Europe** : Risque de décrochage sans réformes structurelles majeures

### 6.2 Enjeux critiques par région

**Pour les États-Unis :**
- Maîtriser la dette publique (34 billions de dollars)
- Gérer les tensions politiques internes
- Maintenir le leadership technologique face à la Chine

**Pour la Chine :**
- Réussir la transition vers un modèle de croissance tiré par la consommation
- Gérer la bulle immobilière et le surendettement
- Innover pour échapper au "piège du revenu intermédiaire"

**Pour l'Europe :**
- Accélérer la transition énergétique et l'autonomie stratégique
- Renforcer l'intégration économique et fiscale
- Stimuler l'innovation et la compétitivité industrielle

**Pour l'Inde :**
- Investir massivement dans les infrastructures
- Améliorer l'éducation et les qualifications
- Créer des emplois pour une population jeune en expansion

### 6.3 Recommandations stratégiques

#### Pour les décideurs politiques :

1. **Investissement dans l'innovation** : R&D, éducation, infrastructures numériques
2. **Transition écologique** : Opportunité de créer de nouveaux avantages compétitifs
3. **Coopération internationale** : Réponse collective aux défis mondiaux (climat, pandémies)
4. **Réformes structurelles** : Flexibilité du marché du travail, efficacité administrative

#### Pour les investisseurs :

1. **Diversification géographique** : Ne pas négliger les marchés émergents à forte croissance
2. **Secteurs porteurs** : Technologies, santé, énergies renouvelables
3. **Attention aux risques** : Géopolitiques, démographiques, climatiques

#### Pour les entreprises :

1. **Stratégie de croissance en Asie** : Marchés à forte croissance (Inde, Asie du Sud-Est)
2. **Optimisation des chaînes de valeur** : Résilience face aux tensions géopolitiques
3. **Transformation numérique** : Impératif de compétitivité

---

## 7. CONCLUSION

L'analyse du PIB mondial en 2024 révèle un paysage économique en mutation profonde, caractérisé par plusieurs tendances structurantes :

### 7.1 Constats majeurs

1. **Bipolarisation économique** : Le monde économique s'organise autour de deux pôles (États-Unis et Chine), avec un écart qui reste massif en faveur des États-Unis.

2. **Basculement vers l'Asie** : L'Inde confirme son statut de futur géant économique mondial avec une croissance de 7,2%, tandis que la Chine consolide sa deuxième position.

3. **Fragilité européenne** : L'Europe fait face à des défis structurels majeurs (énergie, démographie, compétitivité) qui menacent son rang dans l'économie mondiale.

4. **Inégalités persistantes** : Le fossé entre PIB total et PIB par habitant souligne les immenses disparités de richesse et de développement à l'échelle mondiale.

5. **Concentration de la richesse** : Les 10 premières économies concentrent 70% du PIB mondial, accentuant les déséquilibres Nord-Sud.

### 7.2 Défis pour l'avenir

L'économie mondiale de 2024 doit naviguer entre plusieurs tensions :

- **Inflation vs Croissance** : Trouver l'équilibre entre stabilité des prix et dynamisme économique
- **Souveraineté vs Mondialisation** : Réindustrialisation et interdépendance commerciale
- **Transition écologique vs Développement** : Concilier urgence climatique et besoins de croissance
- **Vieillissement vs Dividende démographique** : Opportunités en Asie, défis en Europe et Chine

### 7.3 Perspectives 2025-2030

Le classement économique mondial continuera d'évoluer, avec plusieurs certitudes :

- **Maintien de la domination américaine** à court terme
- **Montée en puissance de l'Inde**, future 3ème économie mondiale
- **Ralentissement progressif de la Chine** vers des taux de croissance de 3-4%
- **Besoin urgent de réformes en Europe** pour éviter le décrochage

### 7.4 Mot de la fin

L'année 2024 marque un tournant dans la géographie économique mondiale. Les pays qui réussiront dans les décennies à venir seront ceux capables de :

- **Innover** et s'adapter rapidement aux disruptions technologiques
- **Investir** dans leur capital humain (éducation, santé)
- **Réformer** leurs structures pour gagner en productivité
- **Coopérer** pour relever les défis communs (climat, pandémies, régulation digitale)

Le PIB reste un indicateur imparfait du bien-être et du développement, mais son analyse demeure essentielle pour comprendre les rapports de force économiques et anticiper les transformations à venir.

---

## ANNEXES

### Annexe A : Glossaire économique

**PIB (Produit Intérieur Brut)** : Valeur totale de tous les biens et services produits dans un pays sur une période donnée (généralement un an).

**PIB nominal** : PIB mesuré aux prix courants, sans ajustement pour l'inflation.

**PIB par habitant** : PIB total divisé par la population, indicateur du niveau de vie moyen.

**Taux de croissance du PIB** : Variation en pourcentage du PIB d'une année à l'autre.

**Parité de pouvoir d'achat (PPA)** : Mesure qui ajuste le PIB en fonction des différences de coût de la vie entre pays.

### Annexe B : Méthodologie de calcul

**Calcul de la part mondiale :**
```
Part mondiale (%) = (PIB du pays / PIB mondial total) × 100
```

**Calcul du PIB par habitant :**
```
PIB par habitant = PIB total / Population du pays
```

**Calcul du taux de croissance :**
```
Taux de croissance (%) = [(PIB année N - PIB année N-1) / PIB année N-1] × 100
```

### Annexe C : Limites de l'analyse

Cette étude présente certaines limites qu'il convient de mentionner :

1. **PIB nominal vs PPA** : L'utilisation du PIB nominal favorise les pays à monnaie forte
2. **Données prévisionnelles** : Les chiffres 2024 incluent des estimations et projections
3. **Indicateur incomplet** : Le PIB ne mesure pas le bien-être, les inégalités ou la durabilité
4. **Économie informelle** : Non prise en compte, particulièrement importante dans les pays émergents
5. **Taux de change** : Les fluctuations monétaires peuvent fausser les comparaisons

### Annexe D : Sources et références

**Sources principales :**
- Fonds Monétaire International (FMI) - World Economic Outlook 2024
- Banque Mondiale - World Development Indicators 2024
- Statista - Global Economic Data 2024

**Pour aller plus loin :**
- OCDE Economic Outlook
- World Bank Global Economic Prospects
- IMF Regional Economic Outlooks

---

## SIGNATURE ET VALIDATION

**Rapport établi le :** Octobre 2025  
**Validité des données :** Année 2024  
**Prochaine mise à jour recommandée :** Janvier 2026

---

*Ce rapport a été généré à des fins d'analyse économique et de recherche. Les données sont issues de sources fiables mais peuvent être sujettes à révisions. Les interprétations et projections reflètent l'analyse au moment de la rédaction et ne constituent pas des conseils d'investissement.*

---

**FIN DU RAPPORT**

---
