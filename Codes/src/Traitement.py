import pandas as pd 
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

#chargement du dataset
data = pd.read_csv('../../Assets/Dataset/spam.csv')
data.head()

#enlever les placement vide 
data.dropna(inplace=True)

#enlevement du doublement
data.drop_duplicates(inplace=True)

#changement de "ham" en "not spam"
data['Category'] = data['Category'].replace(["ham","spam"],["Not spam","Spam"])

#nettoyage
def nettoyage(texte):
    if not isinstance(texte, str):
        return ""
    
    # Minuscules
    texte = texte.lower()
    
    # Tokenisation
    mots = word_tokenize(texte)
    
    # Suppression ponctuation
    table_ponctuation = str.maketrans('', '', string.punctuation)
    mots = [m.translate(table_ponctuation) for m in mots]
    mots = [m for m in mots if m != '']
    
    # Suppression des Stop Words
    mots_vides = set(stopwords.words('english'))
    mots = [m for m in mots if m not in mots_vides]
    
    # Lemmatisation
    lemmatiseur = WordNetLemmatizer()
    mots_propres = [lemmatiseur.lemmatize(m) for m in mots]
    
    return " ".join(mots_propres)
#application sur dataset
data['mess_clean'] = data['Message'].apply(nettoyage)

#resultat
print(data[['Message', 'mess_clean', 'Category']].head(10))

