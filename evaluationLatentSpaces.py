# evaluate logistic regression on encoded input
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import keras
import constantes
import functions
from vaePourClassif import Sampling

train_data =  pd.read_csv("binary_train_dataset.csv") 
train_labels = train_data[constantes.MULTICLASS_TARGET_COL]
test_data = pd.read_csv("binary_test_dataset.csv") 
test_labels=test_data[constantes.MULTICLASS_TARGET_COL]
train_data = train_data.drop(columns=constantes.MULTICLASS_TARGET_COL)
test_data = test_data.drop(columns=constantes.MULTICLASS_TARGET_COL)   
    
    # Combine the training and testing datasets for preprocessing
combined_data = pd.concat([train_data, test_data], axis=0)
numerical_columns = combined_data.select_dtypes(include=[np.number]).columns

    # Initialize the MinMaxScaler
scaler = MinMaxScaler()

    # Scale the numerical columns (excluding "label")
combined_data[numerical_columns] = scaler.fit_transform(
        combined_data[numerical_columns]
    )

 # Split the combined dataset back into training and testing datasets
train_data = combined_data[: len(train_data)]
test_data = combined_data[len(train_data) :]


    # Convert the data to float64
train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)

train_data, client_data, train_labels, client_labels = train_test_split(train_data, train_labels, train_size=0.3,stratify=train_labels, random_state=42)




encoder = keras.saving.load_model("vae_encoder_sauv.keras", custom_objects={"Sampling": Sampling})


encoderAe = keras.saving.load_model("encoder.keras")

# Limiter à 10 000 exemples pour l'encodage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = test_data.copy()
df["label"] = test_labels.values

# Séparer BENIGN et ATTACK
benign_df = df[df["label"] == "Benign"]
attack_df = df[df["label"] != "Benign"]

# Vérifier qu'on a au moins 10 000 de chaque
print("BENIGN samples:", len(benign_df))
print("ATTACK samples:", len(attack_df))

# Échantillonnage aléatoire de 10 000 chacun
benign_sample = benign_df.sample(n=10000, random_state=42)
attack_sample = attack_df.sample(n=10000, random_state=42)

# Fusionner et mélanger
subset_df = pd.concat([benign_sample, attack_sample]).sample(frac=1, random_state=42)

# Extraire les données et les labels
subset = subset_df.drop(columns=["label"])
subset_labels = subset_df["label"]

# Vérification
print("Après échantillonnage équilibré :")
print(subset_labels.value_counts())

# Encodage des labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(subset_labels)


# encode the train data
x_VAE = encoder.predict(subset)
x_AE = encoderAe.predict(subset)

functions.latentSpace_UMAP(x_VAE ,encoded_labels )

