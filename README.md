# Arquitetura da Rede Multi Layer Perceptron 

A rede possui três camadas:

1. Camada de Entrada

2. Camada Oculta (ReLU)

3. Camada de Saída (Sigmoid)

| Arquitetura   | Valores | 
|--------------|----------|
| Input Size   | 11       |
| Hidden Size  | 64       |
| Output Size  | 1        | 

## Equações:

z1 = X · W1 + b1
a1 = ReLU(z1)


z2 = a1 · W2 + b2
a2 = Sigmoid(z2)

Onde:

W1, W2 = pesos

b1, b2 = bias

a1 = ativação da camada oculta

a2 = saída final (probabilidade da respectiva classe)

# Funções de Ativação Usadas

## ReLU (Hidden Layer)
ReLU(x) = max(0, x)

## Sigmoid (Output Layer)
Sigmoid(x) = 1 / (1 + exp(-x))

## Gradientes:
ReLU'(x) = 1 se x > 0, senão 0

Sigmoid'(x) = x * (1 - x)

# Função de Perda 
Binary Cross-Entropy: 

Loss = -mean( y·log(p) + (1-y)·log(1-p) )

# Adicionais

**EXPERIMENTOMLP1.PY:** Utilização de K-Fold Cross-Validation, Mini-Batches durante a fase de treinamento e shuffle dos dados para garantia da aleatoriedade, além de L2 Regularization nos pesos. 

**EXPERIMENTOMLP2.PY:** Salvamento da MLP pré-treinada no arquivo "ExperimentoMLP1.py" como arquivo .pkl e retreinamento, novamente, em uma nova base de dados, isto é, aplicação de Aprendizado Online para melhora da performance do modelo de redes neurais.  

# Resultados por Experimento 

| Experimentos | Acurácia  | 
|--------------|----------|
| Experimento 1| 80.26%   |
| Experimento 2| 82.50%   |
| Experimento 3| 89.13%   |
| Experimento 4| 89.57%   |

<img width="377" height="375" alt="image" src="https://github.com/user-attachments/assets/510a32b5-de33-4b49-a960-f55003de11d6" />

# Referências 

1. Dataset: https://www.kaggle.com/datasets/nvarisha/heart-attack-data-analysis/data

2. Dataset: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

3. BRAGA, Antônio de Pádua; LUDERMIR, Teresa Bernarda; CARVALHO, André Carlos Ponce de Leon Ferreira de. Redes neurais artificiais: teoria e aplicações. 2000.
