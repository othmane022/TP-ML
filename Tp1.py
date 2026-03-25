
#INTRODECTION AU DATASET
#verfer le dataset
import pandas as pd
df = pd.read_csv("train.csv")
#les types des donnees
print(df.dtypes)
#affiche les 5 linge premaire
print(df.head())
#afiche les valeur manquant a chaque linge
print(df.isnull().sum())
#Nettoyage des Données
#Gestion des valeurs manquantes soit moyenne ou mediane
# df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Age']=df['Age'].fillna(df['Age'].median())
#variables catégorielles
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
#la supprimer
df.drop(columns=['Cabin'], inplace=True)
#Gestion des valeurs aberrantes
#Utilisez des boxplots pour détecterimport seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df['Age'])
plt.show()
#Remplacez ou supprimez les valeurs aberrantes,
import numpy as np
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['Age'] = np.where(df['Age'] < lower, df['Age'].median(), df['Age'])
df['Age'] = np.where(df['Age'] > upper, df['Age'].median(), df['Age'])
df = df[(df['Age'] >= lower) & (df['Age'] <= upper)]
#Transformation des Données--->encodage des varaibles categorielles
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df['Pclass'] = df['Pclass'].map({1: 3, 2: 2, 3: 1})
#Normalisation et Standardisation
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
#Standardisation (Z-score)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
#Visualisation avec Histogramme
import matplotlib.pyplot as plt

df['Age'].hist()
plt.title("Distribution après transformation")
plt.show()
#partier 4 : Gestion des Données Déséquilibrées
#Identification du déséquilibre
df['Survived'].value_counts()
#Techniques de gestion(Sous-échantillonnage (Undersampling),Sur-échantillonnage (Oversampling - SMOTE))
from sklearn.utils import resample

df_majority = df[df['Survived'] == 0]
df_minority = df[df['Survived'] == 1]

df_majority_downsampled = resample(df_majority,replace=False,n_samples=len(df_minority),random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
#Sur-échantillonnage (Oversampling - SMOTE)
from imblearn.over_sampling import SMOTE

X = df.drop('Survived', axis=1)
y = df['Survived']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
