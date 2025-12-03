**Thématique**

**Cybersécurité :** Détection d'intrusions réseau, analyse de phishing ou de spams.

**1\. Informations Générales & Contexte**

- **Nom du Dataset :** _Dataset Phishing Domain Detection_ (ou _Phishing Domain URL Dataset_).
- **Auteur/Source :** Mis à disposition par **MichelleVP** (Michelle Velice Patricia) sur la plateforme Kaggle.
- **Contexte Métier :** La détection traditionnelle de phishing repose sur des "listes noires" (blacklists) qui sont souvent réactives et lentes à se mettre à jour. Ce dataset permet une approche proactive via le Machine Learning : analyser la _structure_ d'une URL pour prédire sa dangerosité avant même qu'elle ne soit signalée.

**2\. Métadonnées (Dictionnaire des Données)**

Le dataset se présente sous la forme d'un fichier CSV tabulaire. Voici le détail probable que vous confirmerez avec votre code (df.info()) :

- **Volume estimé :** Environ **58 000 à 88 000 lignes** (instances d'URL) et **112 colonnes** (variables).
- **Type de données :** Majoritairement numériques (entiers int ou flottants float) pour les caractéristiques, et une variable catégorielle (texte ou entier) pour la cible.

**3\. Description Détaillée des Features (Variables Explicatives)**

Les variables ne sont pas le contenu de la page web (texte, images), mais des **propriétés lexicales et techniques** extraites de l'URL elle-même. Elles sont regroupées en plusieurs catégories :

**A. Caractéristiques basées sur la quantité (qty_...)**

Ces variables comptent le nombre d'occurrences de caractères spécifiques dans l'URL. Les pirates utilisent souvent ces caractères pour masquer la véritable destination ou obfusquer le code.

- **Exemples :** qty_dot_url (nombre de points), qty_hyphen_url (tirets), qty_slash_url (barres obliques), qty_questionmark_url (points d'interrogation), qty_at_url (@), qty_tld_url (nombre d'extensions de domaine).
- **Interprétation :** Une URL légitime a rarement 5 points ou 3 arobases. Une quantité élevée est souvent suspecte.

**B. Caractéristiques de Longueur**

- **Variables :** length_url (longueur totale), domain_length (longueur du nom de domaine), directory_length (longueur du chemin d'accès).
- **Interprétation :** Les URLs de phishing sont souvent très longues pour cacher le domaine réel sur les écrans mobiles.

**C. Caractéristiques Techniques & Réseau**

- **domain_in_ip :** Indique si le domaine est remplacé par une adresse IP (ex: <http://192.168.1.1/>...). C'est un signe très fort de phishing.
- **server_client_response :** Temps de réponse du serveur.
- **tls_ssl_certificate :** Présence ou non d'un certificat de sécurité (les pirates utilisent de plus en plus le HTTPS pour tromper les victimes).
- **email_in_url :** Indique si une adresse email apparaît en clair dans l'URL.

**4\. La Variable Cible (Target)**

- **Nom de la colonne :** Généralement nommée phishing, status ou label.
- **Type :** Binaire (Classification).
- **Valeurs :**
  - 0 (ou legitimate) : L'URL est saine.
  - 1 (ou phishing) : L'URL est frauduleuse.
- **Équilibre :** Ce dataset est réputé pour être **équilibré** (environ 50% de phishing / 50% de légitime), ce qui facilite l'apprentissage sans avoir besoin de techniques complexes de rééchantillonnage (SMOTE).
