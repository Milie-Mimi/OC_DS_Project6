''' Fonctions de preprocessing et d'extraction des features image
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing image
from PIL import Image, ImageOps, ImageFilter
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# --------------------------------------------------------------------
# -- HISTOGRAMME DE REPARTITION DES PIXELS
# --------------------------------------------------------------------            

def img_histogramme(image, titre, density=False):
    '''Fonction qui affiche l'image, l'histogramme de répartition des 
    pixels et l'histogramme cumulé

    Arguments:
    --------------------------------
    image: image, obligatoire
    titre: str: titre au dessus de l'image, obligatoire
    density: bool: 
    
    return:
    --------------------------------
    None
    '''
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.title(titre, fontsize=30)
    plt.imshow(image, cmap='gray')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(132)
    plt.title('Histogramme de répartition des pixels', fontsize=30)
    hist, bins = np.histogram(np.array(image).flatten(), bins=256, density=density)
    plt.bar(range(len(hist[0:255])), hist[0:255])
    plt.xlabel('Niveau de gris', fontsize=25)
    plt.ylabel('Nombre de pixels', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.subplot(133)
    plt.title('Histogramme cumulé des pixels', fontsize=30)
    plt.hist(np.array(image).flatten(), bins=range(256), cumulative=True, density=density)
    plt.xlabel('Niveau de gris', fontsize=25)
    plt.ylabel('Fréquence cumulée de pixels', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.show()


# --------------------------------------------------------------------
# -- PREPROCESSING IMAGES
# --------------------------------------------------------------------            
    
    
def picture_preprocessing(picture_path):
    '''Fonction regroupant les divers traitements à effectuer 
    sur les images: redimensionnement 224 * 224, passage en gris, 
    correction du contraste et réduction du bruit avec filtre 
    non local means

    Arguments:
    --------------------------------
    picture_path: str, chemin d'accès à l'image
    
    return:
    --------------------------------
    image preprocessée
    '''
    # Chargement de l'image comme matrice de pixels
    img = Image.open(picture_path)
    # Redimensionnement
    newsize = (224, 224)
    img = img.resize(newsize)
    # Passage en gris
    img_grey = img.convert("L")
    # Correction du contraste
    img_contrast = ImageOps.equalize(img_grey)
    # Réduction du bruit par filtre non-local means
    img_contrast_noise_nlm = cv2.fastNlMeansDenoising(np.array(img_contrast), None, 5, 8, 21)
    return img_contrast_noise_nlm


# --------------------------------------------------------------------
# -- ACP, TSNE, KMEANS => ARI
# --------------------------------------------------------------------     

def etiquette_v(ax, espace=5):
    '''Ajoute les étiquettes en haut de chaque barre sur un barplot vertical.

    Arguments:
    --------------------------------
    ax: (matplotlib.axes.Axes): objet matplotlib contenant les axes
    du plot à annoter.
    espace: int: distance entre les étiquettes et les barres
    
    return:
    --------------------------------
    None
    '''

    # Pour chaque barre, placer une étiquette
    for rect in ax.patches:
        # Obtenir le placement de X et Y de l'étiquette à partir du rectangle
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Espace entre la barre et le label
        space = espace
        # Alignement vertical
        va = 'bottom'

        # Si la valeur est négative, placer l'étiquette sous la barre
        if y_value < 0:
            # Valeur opposer de l'argument espace
            space *= -1
            # Alignement vertical 'top'
            va = 'top'

        # Utiliser la valeur Y comme étiquette et formater avec 0 décimale
        label = "{:.0f}".format(y_value)

        # Créer l'annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)


# -----------------------
# -- ACP, TSNE, KMEANS, ARI
# -----------------------

def ARI_SIFT(features, n_comp, n_clust, df, col_categ) :
    '''Calcul ARI entre vraies catégories et n° de clusters après
    pré-traitement des images via SIFT

    Arguments:
    --------------------------------
    features: array: matrice des histogrammes, obligatoire
    n_comp: int: nombre de composantes principales, obligatoire
    n_clust: int: nombre de clusters pour le kmeans (= nombre de catégorie), obligatoire
    df: dataframe: tableau en entrée, obligatoire
    col_categ: str: nom de la colonne à prédire, obligatoire
    dataframe: DataFrame: dataframe en entrée avec les n° de clusters
    
    return:
    --------------------------------
    ARI: ARI entre vraies catégories et n° de clusters pour modèles
    X_tsne: matrice TSNE
    cls.labels: clusters après K-means'''
    

    # Liste des catégories
    categ_list = list(set(df[col_categ]))
    # Affectation indice de catégories
    y_cat_num = [(1-categ_list.index(df.iloc[i][col_categ])) for i in range(len(df))]
    
    # Standardisation des données
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    # ACP
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(features_std)

    # t-SNE
    tsne = manifold.TSNE(n_components=2,
                         perplexity=30,
                         n_iter=2000,
                         random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = KMeans(n_clusters=n_clust,
                 init='k-means++',
                 random_state=42)
    cls.fit(X_tsne)
    
    # Ajout des clusters dans la dataframe original
    datataframe = df.copy()
    datataframe['Clusters'] = cls.labels_
    datataframe['Clusters_str'] = ["Cluster_" + str(row) for row in datataframe['Clusters']]
    
    # Distribution des produits par cluster
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x='Clusters_str',
                       data=datataframe,
                       order=datataframe['Clusters_str'].value_counts().index)
    plt.title('Distribution des produits par cluster',
              fontweight='bold',fontsize=15)
    plt.xlabel("", fontsize=12)
    etiquette_v(ax, 2)
    
    ARI = np.round(adjusted_rand_score(y_cat_num, cls.labels_),4)
    
    
    return ARI, X_tsne, cls.labels_, datataframe


# -----------------------
# -- TSNE, KMEANS, ARI
# -----------------------

def ARI_CNN(features, n_clust, df, col_categ) :
    '''Calcul ARI entre vraies catégories et n° de clusters après
    pré-traitement des images via CNN

    Arguments:
    --------------------------------
    features: array: matrice des histogrammes, obligatoire
    n_clust: int: nombre de clusters pour le kmeans (= nombre de catégorie), obligatoire
    df: dataframe: tableau en entrée, obligatoire
    col_categ: str: nom de la colonne à prédire, obligatoire
    
    return:
    --------------------------------
    ARI: ARI entre vraies catégories et n° de clusters pour modèles
    X_tsne: matrice TSNE
    cls.labels: clusters après K-means
    dataframe: DataFrame: dataframe en entrée avec les n° de clusters'''
    

    # Liste des catégories
    categ_list = list(set(df[col_categ]))
    # Affectation indice de catégories
    y_cat_num = [(1-categ_list.index(df.iloc[i][col_categ])) for i in range(len(df))]
    

    # t-SNE
    tsne = manifold.TSNE(n_components=2,
                         perplexity=30,
                         n_iter=2000,
                         random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = KMeans(n_clusters=n_clust,
                 init='k-means++',
                 random_state=42)
    cls.fit(X_tsne)
    
    # Ajout des clusters dans la dataframe original
    datataframe = df.copy()
    datataframe['Clusters'] = cls.labels_
    datataframe['Clusters_str'] = ["Cluster_" + str(row) for row in datataframe['Clusters']]
    
    # Distribution des produits par cluster
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(x='Clusters_str',
                       data=datataframe,
                       order=datataframe['Clusters_str'].value_counts().index)
    plt.title('Distribution des produits par cluster',
              fontweight='bold',fontsize=15)
    plt.xlabel("", fontsize=12)
    etiquette_v(ax, 2)
    
    ARI = np.round(adjusted_rand_score(y_cat_num, cls.labels_),4)
    
    
    return ARI, X_tsne, cls.labels_, datataframe