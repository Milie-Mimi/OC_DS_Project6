''' Fonctions de manipulation des dataframes
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------
# -- SHAPE & NAN
# --------------------------------------------------------------------


def shape_total_nan(df):
    '''Fonction qui retourne le nombre de lignes,
    de variables, le nombre total de valeurs manquantes et
    le pourcentage associé
    
    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    
    
    return:
    --------------------------------
    None'''
    
    
    missing = df.isna().sum().sum()
    missing_percent = round(missing
                            / (df.shape[0] * df.shape[1])
                            * 100,
                            2)

    print(f"Nombre de lignes: {df.shape[0]}")
    print(f"Nombre de colonnes: {df.shape[1]}")
    print(f"Nombre total de NaN du dataset: {missing}")
    print(f"% total de NaN du dataset: {missing_percent}%")


# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES
# --------------------------------------------------------------------


def describe_variables(data):
    ''' Fonction qui prend un dataframe en entrée, et retourne un
    récapitulatif qui contient le nom des variables, leur type, un
    exemple de modalité, le nombre total de lignes, le nombre et
    pourcentage de valeurs distinctes, le nombre et pourcentage de
    valeurs non manquantes et de valeurs manquantes (NaN) et les
    principales statistiques pour les variables numériques (moyenne,
    médiane, distribution, variance, écart type, minimum, quartiles et
    maximum)
    
    Arguments:
    --------------------------------
    data: dataframe: tableau en entrée, obligatoire
    
    
    return:
    --------------------------------
    dataframe qui décrit les variables'''

    # Choix du nom des variables à afficher
    df = pd.DataFrame(columns=[
        'Variable name', 'Variable type', 'Example', 'Raws', 'Distinct',
        '% distinct', 'Not NaN', '% Not NaN', 'NaN', '% NaN', 'Mean',
        'Median', 'Skew', 'Kurtosis', 'Variance', 'Std', 'Min', '25%',
        '75%', 'Max'
    ])

    # Pour chaque colonne du dataframe
    for col in data.columns:

        # Définition des variables
        # type de la variable (object, float, int...)
        var_type = data[col].dtypes
        # premier élément notNA
        example = data[data[col].notna()][col].iloc[0]
        # nombre total de lignes
        nb_raw = len(data[col])
        # nombre de valeurs non manquantes
        count = len(data[col]) - data[col].isna().sum()
        # % de valeurs non manquantes
        percent_count = round(data[col].notnull().mean(), 4)*100
        # nombre de modalités que peut prendre la variable
        distinct = data[col].nunique()
        # % de valeurs distinctes
        percent_distinct = round(data[col].nunique()/len(data[col]), 4)
        percent_distinct = percent_distinct * 100
        # nombre de valeurs manquantes
        missing = data[col].isna().sum()
        # % de valeurs manquantes
        percent_missing = round(data[col].isna().mean(), 4)*100

        # Pour les var de type 'int' ou 'float': on remplit toutes les col
        if var_type == 'int32' or var_type == 'int64' or var_type == 'float':
            df = pd.concat([df, pd.DataFrame([[col, var_type, example, nb_raw,
                                               distinct, percent_distinct,
                                               count,
                                               percent_count,
                                               missing,
                                               percent_missing,
                                               round(data[col].mean(), 2),
                                               round(data[col].median(), 2),
                                               round(data[col].skew(), 2),
                                               round(data[col].kurtosis(), 2),
                                               round(data[col].var(), 2),
                                               round(data[col].std(), 2),
                                               round(data[col].min(), 2),
                                               round(data[col].quantile(0.25),
                                                     2),
                                               round(data[col].quantile(0.75),
                                                     2),
                                               data[col].max()]],
                                             columns=['Variable name',
                                                      'Variable type',
                                                      'Example',
                                                      'Raws',
                                                      'Distinct',
                                                      '% distinct',
                                                      'Not NaN',
                                                      '% Not NaN',
                                                      'NaN',
                                                      '% NaN',
                                                      'Mean',
                                                      'Median',
                                                      'Skew',
                                                      'Kurtosis',
                                                      'Variance',
                                                      'Std',
                                                      'Min',
                                                      '25%',
                                                      '75%',
                                                      'Max'])])

            # Pour les variables d'un autre type: on ne remplit que
            # les variables de compte

        else:
            df = pd.concat([df, pd.DataFrame([[col, var_type, example,
                                               nb_raw, distinct,
                                               percent_distinct,
                                               count,
                                               percent_count, missing,
                                               percent_missing,
                                               '', '', '', '', '', '',
                                               '', '', '', '']],
                                             columns=['Variable name',
                                                      'Variable type',
                                                      'Example',
                                                      'Raws',
                                                      'Distinct',
                                                      '% distinct',
                                                      'Not NaN',
                                                      '% Not NaN',
                                                      'NaN',
                                                      '% NaN',
                                                      'Mean',
                                                      'Median',
                                                      'Skew',
                                                      'Kurtosis',
                                                      'Variance',
                                                      'Std',
                                                      'Min',
                                                      '25%',
                                                      '75%',
                                                      'Max'])])

    return df.reset_index(drop=True)


# --------------------------------------------------------------------
# -- SPLIT COLONNE EN PLUSIEURS COLONNES
# --------------------------------------------------------------------


def split_and_add_columns(df, var, sep):
    ''' Fonction qui prend un dataframe en entrée, split une colonne
    en plusieurs colonnes en fonction d'un séparateur
    
    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    var: str: nom de la colonne à splitter, obligatoire
    sep: str: séparateur qui permet de savoir à partir d'où
    effectuer le split
    
    return:
    --------------------------------
    dataframe avec des colonnes supplémentaires'''

    # Split des catégories en x colonnes
    df_cat = df[var].str.split(sep, expand=True)
    len_df_cat = len(df_cat.columns.tolist())

    # Modification automatique du nom des colonnes
    for i in range(0, len_df_cat):
        df_cat = df_cat.rename(columns={df_cat.columns[i]:
                                        'Cat_' + str(i)})

    # Assemblage des 2 dataframes
    df_new = pd.concat([df, df_cat], sort=False, axis=1)

    # Nettoyage des ["
    df_new.replace('\\["', '', regex=True, inplace=True)
    df_new.replace('"\\]', '', regex=True, inplace=True)

    return df_new