''' Fonctions Machine Learning
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score



# --------------------------------------------------------------------
# -- PLOT EBOULIS DES VALEURS PROPRES (ACP)
# --------------------------------------------------------------------            

def display_scree_plot(pca):
    '''Fonction qui affiche l'éboulis des valeurs propres.

    Arguments:
    --------------------------------
    pca: acp, obligatoire
    
    return:
    --------------------------------
    None
    '''
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


# --------------------------------------------------------------------
# -- PLOT TSNE
# --------------------------------------------------------------------  


def TSNE_visu(X_tsne, y_cat_num, labels) :
    '''Représentation graphique du Tsne.

    Arguments:
    --------------------------------
    X_tsne: matrice TSNE, obligatoire
    y_cat_num: affectation indice de catégories, obligatoire
    labels: liste des catégories, obligatoire
    
    return:
    --------------------------------
    None
    '''
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=labels, loc="best", title="Categorie")
    plt.title('Représentation des produits par catégories')
    
    plt.show()

    
def TSNE_visu_fct(X_tsne, labels, ARI, df, col_categ) :
    '''Visualisation du t-SNE selon les vraies catégories et selon les clusters.

    Arguments:
    --------------------------------
    X_tsne: matrice après TSNE, obligatoire
    labels: liste des clusters, obligatoire
    ARI: float: Adjusted Rand Score
    df: dataframe: tableau en entrée, obligatoire
    col_categ: str: nom de la colonne à prédire, obligatoire
    
    return:
    --------------------------------
    None
    '''
    
    # Liste des catégories
    categ_list = list(set(df[col_categ]))
    # Affectation indice de catégories
    y_cat_num = [(1-categ_list.index(df.iloc[i][col_categ])) for i in range(len(df))]   
    
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=categ_list, loc="best", title="Categorie")
    plt.title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des produits par clusters')
    
    plt.show()
    print("ARI : ", ARI)    