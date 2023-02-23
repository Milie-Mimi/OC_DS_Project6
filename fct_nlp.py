''' Fonctions de traitement du language naturel
'''

import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Natural Language Processing
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer, PorterStemmer
import collections

# BOW
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Word2Vec
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from gensim.models import Word2Vec
import gensim

# BERT
# import tensorflow_hub as hub
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Bert Huggingface
import os
import transformers
from transformers import *

os.environ["TF_KERAS"]='1'

# Bert Tensorflow Hub
import tensorflow_hub as hub
import tensorflow_text 

# USE
import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.keras
from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Bert
import transformers
from transformers import *

os.environ["TF_KERAS"]='1'

import tensorflow_hub as hub


# --------------------------------------------------------------------
# -- NOMBRE DE TOKENS
# --------------------------------------------------------------------


def nb_tokens(txt):
    '''Renvoie le nombre de tokens et de tokens uniques.

    Arguments:
    --------------------------------
    txt: liste: liste de tokens, obligatoire
    
    return:
    --------------------------------
    None
    '''
    print("Nombre de tokens: {}".format(len(txt)))
    print("Nombre de tokens uniques: {}".format(len(set(txt))))


# --------------------------------------------------------------------
# -- PREPROCESSING TEXTE
# --------------------------------------------------------------------    


def clean_txt(text,
              join=True,
              lemm_or_stemm=None,
              stop_words="english",
              eng_words=None,
              min_len_word=2,
              only_alpha=True,
              extra_words=None,
              list_rare_words=None):
    
    '''Natural Language Processing basic function.

    positional arguments:
    ------------------------
    text: str: text in a str format to process

    opt args:
    ------------------------
    join : bool : if True, return a string else return the list of tokens
    lemm_or_stemm : str : if lemm do lemmentize else stemmentize
    stop_words : str: language of the stopwords
    eng_words : list : list of english words
    min_len_word : int : the minimum length of words to keep
    only_alpha : bool : if True, exclude all tokens with a numeric character
    extra_words : list : words to exclude and consider as stopwords
    list_rare_words : list : list of rare words to exclude

    return:
    ------------------------
    a list of tokens'''

    # extra_words
    if not extra_words:
        extra_words = []

    # eng_words
    if not eng_words:
        eng_words = []

    # list_rare_words
    if not list_rare_words:
        list_rare_words = []

    # lower and strip
    doc = text.lower().strip()

    # Tokenize
    # tokenizer = RegexpTokenizer("[A-Za-z0-9]\w+")
    # tokenizer = RegexpTokenizer("[A-Za-z]\w+")
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # Remove stopwords
    # stop_words = list(set(stopwords.words(stop_words)))
    stop_words = set(stopwords.words(stop_words))
    clean_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # Drop extra_words tokens
    extra_w = [w for w in clean_tokens_list if w not in extra_words]

    # No rare tokens
    non_rare_tokens = [w for w in extra_w if w not in list_rare_words]

    # Keep only len word > N
    more_than_N = [w for w in non_rare_tokens if len(w) >= min_len_word]

    # Keep only alpha not num
    if only_alpha == True:
        alpha_num = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_num = more_than_N

    # Stemm or Lemm
    if lemm_or_stemm == "lemm":
        trans = WordNetLemmatizer()
        trans_txt = [trans.lemmatize(i) for i in alpha_num]
    elif lemm_or_stemm == "stemm":
        trans = PorterStemmer()
        trans_txt = [trans.stem(i) for i in alpha_num]
    else:
        trans_txt = alpha_num

    # English words
    if eng_words:
        engl_txt = [i for i in trans_txt if i in eng_words]
    else:
        engl_txt = trans_txt
        
    # Return a list or a string
    if join:
        return " ".join(engl_txt)

    return engl_txt


# ------------------------------------------------------------------------------------------------------------------------------
# -- BAG OF WORDS: COUNTVECTORIZER / TF-IDF
# ------------------------------------------------------------------------------------------------------------------------------


def bag_of_word(df, df_index, feature_fit, feature_trans, model):
    '''Extraction des features texte, vectorisation et visualisation
    du résultat au format dataframe.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    df_index: str: nom de la colonne à mettre en index, obligatoire
    feature_fit: str: nom de la colonne où extraire les features, obligatoire
    feature_trans: str: nom de la colonne où à vectoriser, obligatoire
    model: fct: fonction au format CountVectorizer() par exemple, obligatoire 
    
    
    return:
    --------------------------------
    Dataframe
    '''
    # Initialisation
    mod = model
    # Extraction features du texte
    mod_fit = mod.fit(df[feature_fit])
    # Vectorisation (encodage du document)
    mod_transform = mod.transform(df[feature_trans])
    feat_names = mod.get_feature_names_out()
    #display(mod.vocabulary_)
    # Format DataFrame
    df = pd.DataFrame(mod_transform.toarray(),
                      index=df[df_index],
                      columns=feat_names) / len(feat_names)
    return df


# --------------------------------------------------------------------
# -- ACP, TSNE, KMEANS, ARI
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

def ARI_BOW(features, n_comp, n_clust, df, col_categ) :
    '''Calcul ARI entre vraies catégories et n° de clusters pour modèles
    de type Bag of Words

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



# ------------------------------------------------------------------------------------------------------------------------------
# -- WORD/SENTENCE EMBEDDING
# ------------------------------------------------------------------------------------------------------------------------------

# -----------------------
# -- WORD2VEC
# -----------------------

def W2v(df, col_vocab, vector_size, window, min_count, epochs, maxlen):
    
    # Liste de liste de tokens
    list_tokens = df[col_vocab].to_list()
    list_tokens = [gensim.utils.simple_preprocess(text) for text in list_tokens]
    
    # Initialisation
    mod_w2v = gensim.models.Word2Vec(min_count=min_count,
                                     window=window,
                                     vector_size=vector_size,
                                     seed=42,
                                     workers=1)
    
    # Construction du vocabulaire à partir d'une séquence de mots
    mod_w2v.build_vocab(list_tokens)
    
    # Entrainement du modèle
    mod_w2v.train(list_tokens,
                  total_examples=mod_w2v.corpus_count,
                  epochs=epochs)
    
    # Vectorisation
    vectors_mod_w2v = mod_w2v.wv
    w2v_words = vectors_mod_w2v.index_to_key
    print("Vocabulary size: %i" % len(w2v_words))
    
    # Tokenization (transforme liste de tokens en séquence de mot)
    print("Fit Tokenizer ...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_tokens)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(list_tokens),
                                maxlen=maxlen,
                                padding='post') 
                                                   
    num_words = len(tokenizer.word_index) + 1
    print("Number of unique words: %i" % num_words)
    
    # Embedding matrix
    print("Create Embedding matrix ...")
    w2v_size = vector_size
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    i=0
    j=0
    
    for word, idx in word_index.items():
        i +=1
        if word in w2v_words:
            j +=1
            embedding_vector = vectors_mod_w2v[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = vectors_mod_w2v[word]
            
    word_rate = np.round(j/i,4)
    print("Word embedding rate : ", word_rate)
    print("Embedding matrix: %s" % str(embedding_matrix.shape))
    
    # Création du modèle
    input=Input(shape=(len(x_sentences),maxlen),dtype='float64')
    word_input=Input(shape=(maxlen,),dtype='float64')  
    word_embedding=Embedding(input_dim=vocab_size,
                             output_dim=w2v_size,
                             weights = [embedding_matrix],
                             input_length=maxlen)(word_input)
    word_vec=GlobalAveragePooling1D()(word_embedding)  
    embed_model = Model([word_input],word_vec)
    
    # Exécution du modèle
    embeddings = embed_model.predict(x_sentences)
    embeddings.shape
    
    return embeddings

# -----------------------
# -- BERT
# -----------------------

# Fonction de préparation des sentences
def bert_inp_fct(sentences_list, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences_list:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot


# Fonction de création des features
def feature_BERT_fct(model, model_type, sentences_list, max_length, b_size, mode='HF') :
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    time1 = time.time()

    for step in range(len(sentences_list)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences_list[idx:idx+batch_size], 
                                                                               bert_tokenizer, max_length)
        
        if mode=='HF' :    # Bert HuggingFace
            outputs = model.predict([input_ids, attention_mask, token_type_ids], batch_size=batch_size_pred)
            last_hidden_states = outputs.last_hidden_state

        if mode=='TFhub' : # Bert Tensorflow Hub
            text_preprocessed = {"input_word_ids" : input_ids, 
                                 "input_mask" : attention_mask, 
                                 "input_type_ids" : token_type_ids}
            outputs = model(text_preprocessed)
            last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    time2 = np.round(time.time() - time1,0)
    print("temps traitement : ", time2)
     
    return features_bert, last_hidden_states_tot

# -----------------------
# -- USE
# -----------------------

def feature_USE_fct(sentences, b_size, embed) :
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features

# -----------------------
# -- TSNE, KMEANS, ARI
# -----------------------

def ARI_embedding(features, n_clust, df, col_categ) :
    '''Calcul ARI entre vraies catégories et n° de clusters après
    pré-traitement des images via word/sentence embedding

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
    cls.labels: clusters après K-means'''
    

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