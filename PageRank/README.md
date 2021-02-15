# Description

Les fichiers `pagerank_*.py` sont l'implémentation des algorithmes Topology-driven PageRank, Basic Data-driven PageRank, 
Pull-based PageRank et Pull-Push-based PageRank mentionnés dans https://papers-gamma.link/static/memory/pdfs/191-pagerank_pushpull.pdf.

# Jeu de données

Télécharger les 4 fichiers suivants sur http://cfinder.org/wiki/?n=Main.Data#toc1.

**Wikipedia: Network of pages, Page categories, Category hierarchy**
- dirLinks.zip
- pageNum2Name.zip
- pageCategList.zip
- categNum2Name.zip

Décompressez-les dans le répertoire `/data`.

# Exécution

Exécuter les programmes `pagerank_*.py` avec le paramètre de la valeur alpha (probabilité de téléportation, 0 < alpha < 1) comme la commande
`python pagerank_*.py <alpha-value>`.  

Il est nécessaire que l'environnement d'exécution possède au moins un RAM de 8Go.

# Résultats

Les fichiers dans `/results` sont les résultats de sortie de programmes, avec le jeu de données ci-dessus et `alpha = 0.15`