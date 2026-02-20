import numpy as np 
import MLP 
import pickle as pkl 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import time 

start = time.time()

caminho = r'/home/EddyPerondini/Documentos/Mestrado/Aplicações/Bancos de Dados/Heart Attack Data Set.csv'

df_heartattack = pd.read_csv(caminho) 

#Definindo as variáveis X (atributos) e y (target)

X = df_heartattack.iloc[:, :-1].values
y = df_heartattack.iloc[:, -1].values.reshape(-1,1)

#Implementando K-Fold Cross Validation

kf = KFold(n_splits = 5, shuffle = True, random_state = 67)
scaler = StandardScaler()

resultados_acc = []

for i, (train_index, val_index) in enumerate(kf.split(X)): 

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    np.random.seed(67)

    mlp = MLP.MLP(
        input_size = 13, 
        hidden_size = 64, 
        output_size = 1
    )

    losses, accuracies = mlp.train(X_train_scaled, y_train, epochs = 200, learning_rate = 0.001)

    pred_val = mlp.predict(X_val_scaled)
    
    acc_fold = (pred_val == y_val).mean()
    resultados_acc.append(acc_fold)

    print(f'Fold {i+1}, Acurácia: {acc_fold:.4f}')

end = time.time()

intervalo = end - start

#Salvando o modelo em .pkl ===> passando para a etapa de teste num novo dataset 

with open('mlp_HeartAttack.pkl', 'wb') as f: 
    pkl.dump(mlp, f)

print(f'Tempo (s): {intervalo}')
print(f'\nAcurácia Média Final: {np.mean(resultados_acc):.4f}')

#Gerando a visualização da função de perda conforme as épocas 

plt.figure(figsize=(10,8))

plt.plot(losses, color = 'blue', label = 'Perda')
plt.plot(accuracies, color = 'orange', label = 'Acurácia')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.grid(True)

#Gerando a visualização da matriz de confusão

cm = confusion_matrix(y_val, pred_val)

display = ConfusionMatrixDisplay(cm)

display.plot(cmap='coolwarm')

plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()