# Arquitetura da Rede Multi Layer Perceptron 

A rede possui três camadas:

1. Camada de Entrada

2. Camada Oculta (ReLU)

3. Camada de Saída (Sigmoid)

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

Utilização de K-Fold Cross-Validation, Mini-Batches durante a fase de treinamento e shuffle dos dados para garantia da aleatoriedade, além de L2 Regularization nos pesos. 

# Resultados por Experimento 

| Experimentos | Acurácia  | 
|--------------|----------|
| Experimento 1| 80.26%   |
| Experimento 2| 82.50%   |


# Referências 

1. Dataset: https://www.kaggle.com/datasets/nvarisha/heart-attack-data-analysis/data

2. BRAGA, Antônio de Pádua; LUDERMIR, Teresa Bernarda; CARVALHO, André Carlos Ponce de Leon Ferreira de. Redes neurais artificiais: teoria e aplicações. 2000.
