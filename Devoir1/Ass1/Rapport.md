Description du Dataset : Student Performance (UCI)
👨‍🔬 Qui ? (Les Auteurs)
Les données ont été collectées et préparées par le Dr. Paulo Cortez et Mme. Alice Silva. Ils sont tous deux chercheurs au Département des Systèmes d'Information de l'Université du Minho, située à Guimarães, au Portugal. Paulo Cortez est un chercheur bien connu dans le domaine du data mining et de l'apprentissage automatique (machine learning).
📅 Quand ? (La Période)

Collecte des données : Les données ont été recueillies au cours de l'année scolaire 2005-2006.
Publication du dataset : Le dataset et l'article de recherche associé ont été publiés en 2008.

🎯 Pourquoi ? (L'Objectif Initial)
L'objectif principal des auteurs n'était pas seulement de collecter des données, mais de prouver que l'on pouvait utiliser des techniques de data mining (exploration de données) pour prédire la réussite ou l'échec scolaire. Leur but ultime était de créer un système capable d'identifier tôt les étudiants "à risque" (ceux susceptibles d'échouer) afin que l'école puisse intervenir et leur proposer un soutien pédagogique avant qu'il ne soit trop tard.
Ils voulaient répondre à des questions comme :

Les notes passées (G1, G2) sont-elles les seuls bons prédicteurs de la note finale (G3) ?
Quel est l'impact réel des facteurs sociaux (sorties, consommation d'alcool, temps d'étude, soutien familial) sur les notes ?
Peut-on prédire un échec (G3 < 10) en se basant uniquement sur des données démographiques et sociales, sans même connaître les premières notes ?

🌍 Où et Comment ? (Le Contexte de la Collecte)

Où : Les données proviennent de deux écoles secondaires publiques de la région du Minho au Portugal. Les écoles sont identifiées par "GP" (Gabriel Pereira) et "MS" (Mousinho da Silveira).
Comment : La collecte s'est faite par deux moyens :

Questionnaires : Les étudiants ont rempli des questionnaires pour fournir les données démographiques, sociales et liées à leur mode de vie (ex: studytime, goout, Dalc, Walc, famsup, etc.).
Registres scolaires : Les données objectives comme les notes (G1, G2, G3), les absences (absences) et les échecs passés (failures) ont été extraites des bases de données de l'école.


Les deux fichiers : Les auteurs ont collecté ces informations pour deux matières fondamentales : les Mathématiques (student-mat.csv) et la Langue Portugaise (student-por.csv). C'est pour cela que la bibliothèque ucimlrepo les combine (donnant 1044 lignes au lieu de 395 ou 649).

En résumé, ce n'est pas juste un "fichier Excel" ; c'est le résultat d'un projet de recherche de 2008 visant à appliquer le machine learning à l'éducation (un domaine maintenant appelé Educational Data Mining ou EDM).

📊 Informations du Référentiel UCI

Source : UCI Machine Learning Repository, Dataset 320
Contexte : Ces données concernent les résultats des élèves de l'enseignement secondaire dans deux écoles portugaises.
Nombre d'entrées : 1044 étudiants (combinaison des deux matières)
Nombre de variables : 33 colonnes au total (32 features + 1 target)
Valeurs manquantes : Aucune - la base de données est complète


📋 Description des Variables
Variables Démographiques

school : École de l'étudiant (binaire : "GP" - Gabriel Pereira ou "MS" - Mousinho da Silveira)
sex : Sexe de l'étudiant (binaire : "F" - féminin ou "M" - masculin)
age : Âge de l'étudiant (numérique : de 15 à 22 ans)
address : Type d'adresse du domicile (binaire : "U" - urbain ou "R" - rural)
famsize : Taille de la famille (binaire : "LE3" - inférieur ou égal à 3 ou "GT3" - supérieur à 3)
Pstatus : Statut de cohabitation des parents (binaire : "T" - vivant ensemble ou "A" - séparés)

Variables Familiales et Éducatives

Medu : Niveau d'éducation de la mère (numérique : 0 - aucun, 1 - primaire (4ème année), 2 - 5ème à 9ème année, 3 - secondaire ou 4 - supérieur)
Fedu : Niveau d'éducation du père (numérique : même échelle que Medu)
Mjob : Profession de la mère (nominal : "teacher", "health", "services", "at_home", "other")
Fjob : Profession du père (nominal : même catégories que Mjob)
reason : Raison du choix de cette école (nominal : proximité du "home", "reputation" de l'école, préférence pour certains "course" ou "other")
guardian : Tuteur légal de l'étudiant (nominal : "mother", "father" ou "other")

Variables de Soutien et Activités

traveltime : Temps de trajet domicile-école (numérique : 1 - <15 min, 2 - 15 à 30 min, 3 - 30 min à 1 heure, 4 - >1 heure)
studytime : Temps d'étude hebdomadaire (numérique : 1 - <2 heures, 2 - 2 à 5 heures, 3 - 5 à 10 heures, 4 - >10 heures)
failures : Nombre d'échecs passés dans les classes précédentes (numérique : n si 1≤n<3, sinon 4)
schoolsup : Soutien pédagogique supplémentaire (binaire : yes ou no)
famsup : Soutien familial pour les études (binaire : yes ou no)
paid : Cours particuliers payants dans la matière (binaire : yes ou no)
activities : Activités extra-scolaires (binaire : yes ou no)
nursery : A fréquenté l'école maternelle (binaire : yes ou no)
higher : Souhaite poursuivre des études supérieures (binaire : yes ou no)
internet : Accès Internet à la maison (binaire : yes ou no)
romantic : En relation amoureuse (binaire : yes ou no)

Variables Sociales et de Style de Vie

famrel : Qualité des relations familiales (numérique : de 1 - très mauvaise à 5 - excellente)
freetime : Temps libre après l'école (numérique : de 1 - très peu à 5 - beaucoup)
goout : Sorties avec les amis (numérique : de 1 - très peu à 5 - très élevé)
Dalc : Consommation d'alcool en semaine (numérique : de 1 - très faible à 5 - très élevée)
Walc : Consommation d'alcool le week-end (numérique : de 1 - très faible à 5 - très élevée)
health : État de santé actuel (numérique : de 1 - très mauvais à 5 - très bon)

Variables de Performance Scolaire

absences : Nombre d'absences scolaires (numérique : de 0 à 93)
G1 : Note du premier semestre (numérique : de 0 à 20)
G2 : Note du deuxième semestre (numérique : de 0 à 20)
G3 : Note finale (numérique : de 0 à 20) - Variable cible


📊 Principales Conclusions de l'Article Original (2008)
Voici les principales conclusions de l'article original "Using Data Mining to Predict Secondary School Student Performance" par P. Cortez et A. Silva.
1. La conclusion la plus importante : Les notes passées sont reines
La découverte la plus évidente et la plus significative des auteurs est que le meilleur prédicteur de la note finale (G3) est, de loin, la note du deuxième semestre (G2).

Corrélation de +0.91 entre G2 et G3 - une corrélation extrêmement forte.
La note G1 est également un excellent prédicteur (corrélation de +0.82).
Implication : Pour prédire précisément si un étudiant va réussir ou échouer à la fin de l'année, la meilleure information à avoir est sa note la plus récente. Un étudiant qui s'en sort bien à G2 s'en sortira presque certainement bien à G3.

2. Prédire l'échec sans les notes passées
Le défi le plus intéressant pour les auteurs était : peut-on prédire l'échec d'un étudiant tôt dans l'année, avant même d'avoir les notes G1 ou G2 ?
Ils ont donc entraîné des modèles en ignorant délibérément les notes G1, G2 et G3 et en essayant de prédire un échec (failures > 0). Dans ce scénario, de nouveaux facteurs sont devenus les plus importants.
3. Les 5 facteurs sociaux et comportementaux les plus influents
En dehors des notes, les auteurs ont identifié plusieurs autres facteurs qui avaient un impact notable sur les performances :

failures (Échecs passés) : C'est le facteur négatif le plus puissant. Un étudiant qui a déjà échoué à des cours dans le passé est massivement plus susceptible d'échouer à nouveau.
higher (Veut aller à l'université) : L'ambition personnelle était un prédicteur positif très fort. Les étudiants qui ont répondu "oui" (yes) à vouloir poursuivre des études supérieures avaient tendance à avoir de bien meilleures notes, indépendamment d'autres facteurs.
Medu & Fedu (Éducation des parents) : Le niveau d'éducation de la mère (Medu) et du père (Fedu) était un indicateur important. Des parents ayant un niveau d'études supérieur étaient corrélés à de meilleurs résultats pour l'étudiant.
school (L'école) : L'école fréquentée (GP ou MS) avait un impact notable, suggérant qu'une école était globalement plus performante que l'autre.
goout (Sorties avec les amis) : Un niveau élevé de sorties (goout = 4 ou 5) était fortement corrélé à de moins bonnes notes.

Mention spéciale : La consommation d'alcool (Dalc et Walc) était également identifiée comme un facteur négatif, tout comme le temps de trajet (traveltime).
Résumé des conclusions
Les auteurs ont réussi leur objectif. Ils ont prouvé que :

La réussite scolaire est fortement auto-corrélée (les bonnes notes amènent les bonnes notes).
En l'absence de notes, une combinaison de facteurs comportementaux (failures, higher, goout) et socio-démographiques (Medu, school) peut créer un modèle de machine learning (type Arbre de Décision ou SVM) capable d'identifier les étudiants "à risque" avec une bonne précision.


🎓 Usages Courants du Dataset
Ce jeu de données est très populaire dans la communauté de la data science et du machine learning. Il est le plus souvent utilisé pour :

La Régression : Prédire la note finale exacte (G3) en fonction des autres variables.
La Classification : Prédire si un étudiant va réussir (par exemple, G3 >= 10) ou échouer (G3 < 10).
L'Analyse de Facteurs : Comprendre quels facteurs, notamment sociaux (comme Dalc - consommation d'alcool en semaine, ou goout - sorties), ont le plus d'impact sur la réussite scolaire.
