""" Fonctions de manipulation des graphiques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------
# -- ETIQUETTES BARPLOT
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
        

# --------------------------------------------------------------------
# -- DISTRIBUTION DES MODALITES DES VARIABLES CATEGORIELLES
# --------------------------------------------------------------------        
        
        
def categ_distrib_plot(df, liste_categ_col, nrows, ncols):
    '''Barplot de distribution des modalités des variables catégorielles avec le nombre de 
    modalités en titre.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_categ_col: liste: liste des variables catégorielles, obligatoire
    nrows: int: nombre de lignes
    ncols: int: nombre de plots par colonne
    
    return:
    --------------------------------
    None
    '''
    # Distribution des catégories

    fig = plt.figure(figsize=(15, 10))
    for i, c in enumerate(liste_categ_col, 1):
        ax = fig.add_subplot(nrows, ncols, i)
        modalites = df[c].value_counts()
        n_modalites = modalites.shape[0]

        if n_modalites > 15:
            modalites[0:15].plot.bar(color='#0f77ce', edgecolor='black', ax=ax)

        else:
            modalites.plot.bar(color='#0f77ce', edgecolor='black')

        ax.set_title(f'{c} ({n_modalites} modalités)', fontweight='bold')
        label = [item.get_text() for item in ax.get_xticklabels()]
        short_label = [lab[0:10] + '.' if len(lab) > 10 else lab for lab in label]
        ax.axes.set_xticklabels(short_label)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(w_pad=2, h_pad=2)

