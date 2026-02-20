import numpy as np 
import MLP 
import pickle as pkl 
import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_path = "/home/EddyPerondini/Documentos/Mestrado/Aplicações/Bancos de Dados/heart_augmented.csv"

df = pd.read_csv(file_path)

#Aplicando Label Encoding para variáveis categóricas

encoding = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoding.fit_transform(df[col])
        

#Importando o modelo salvo em .pkl 

with open('mlp_HeartAttack.pkl', 'rb') as f:
    mlp_carregado = pkl.load(f)

X = df.loc[:, df.columns != 'target'].values
y = df.loc[:,'target'].values.reshape(-1,1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size = 0.25, random_state = 67)

#Realizando um novo treinamento ===> Online Learning 

np.random.seed(67)

losses, accuracies = mlp_carregado.train(X_train, y_train, epochs = 200, learning_rate = 0.001)

pred = mlp_carregado.predict(X_val)
acc = (pred == y_val).mean()

print(f'Acurácia obtida no novo dataset: {acc:.4f}')

#Visualização das curvas de aprendizado 

plt.figure(figsize=(10,8))

plt.plot(losses, color = 'blue', label = 'Perda')
plt.plot(accuracies, color = 'orange', label = 'Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid(True)

#Visualização da matriz de confusão 

cm = confusion_matrix(y_val, pred)

display = ConfusionMatrixDisplay(cm)

display.plot(cmap='coolwarm')

plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()




