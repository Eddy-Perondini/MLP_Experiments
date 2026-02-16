import numpy as np 
import MLP 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import time 
import matplotlib 
matplotlib.use("TkAgg")

start = time.time()

caminho = r'/home/EddyPerondini/Documentos/Mestrado/Aplicações/Bancos de Dados/Heart Attack Data Set.csv'

df_heartattack = pd.read_csv(caminho) 

#print(df_heartattack.columns)

#Definindo as variáveis X (atributos) e y (target)

X = df_heartattack.iloc[:, :-1].values
y = df_heartattack.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 67)

#Normalizando os dados para aplicar à rede 

mean = X_train.mean(axis=0)
std  = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

#Criando a rede 

np.random.seed(67)

mlp = MLP.MLP(
    input_size = 13, 
    hidden_size = 64, 
    output_size = 1
    )

losses, accuracies = mlp.train(X_train, y_train, epochs = 5000, learning_rate = 0.001)

pred = mlp.predict(X_test)
acc = (pred == y_test).mean()

end = time.time()

intervalo = end - start

print(f'Tempo (s): {intervalo}')
print("Acurácia:", acc)
print("Classes Preditas pelo Modelo:", pred.ravel())
print("Classes Verdadeiras:", y_test.ravel())

#Gerando a visualização da função de perda conforme as épocas 

plt.figure(figsize=(10,8))

plt.plot(losses, color = 'blue', label = 'Perda')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10,8))

plt.plot(accuracies, color = 'orange', label = 'Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

#Gerando a visualização da matriz de confusão

cm = confusion_matrix(y_test, pred)

display = ConfusionMatrixDisplay(cm)

display.plot(cmap='coolwarm')

plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()