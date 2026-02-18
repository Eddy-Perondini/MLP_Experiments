'''
Construção do zero de uma rede Multi Layer Perceptron para fins de estudo. 

---- Características do modelo ----

Entrada - Vetor X = (x1, x2, ..., xn)

Saída (Classificação) - Vetor Y = (y1, y2)

Pesos - Vetor W = (w1, w2, ..., wn)

Função de Ativação (Função que governa a despolarização do neurônio) - f(.) ==> Existem inumeros tipos de função de ativação: linear, sigmoidal, degrau etc... 

Limiar de Despolarização - Theta ==> Limite escalar para permitir a despolarização 

h(x) = \sum w_{i}x_{i}

------------------------------------

Inclusão do gradiente descendente ==> Processo de ajuste dos pesos via minimização da derivada da função de ativação df(x)/dx = 0 

'''

import numpy as np 
from tqdm import tqdm

#Definindo a função de ativação sigmoid e seu gradiente 

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def grad_sigmoid(x): 
    return x * (1 - x)

def relu(x): 
    return np.maximum(0,x)

def grad_relu(x):
    return (x > 0).astype(float)

#Arquitetura da Rede MLP 

class MLP: 
    def __init__(self, input_size, hidden_size, output_size): 

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def foward(self, X): 

        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2 
        self.a2 = sigmoid(self.z2)

        return self.a2 
        
    def backward(self, X, y, output, learning_rate, lambda_l2 = 0.001):

        delta2 = output - y

        delta1 = delta2.dot(self.W2.T) * grad_relu(self.a1)

        #Aplicando gradientes de pesos ===> buscando L2 Regularization 

        grad_W2 = self.a1.T.dot(delta2)
        grad_W1 = X.T.dot(delta1)

        #Atualização dos pesos c/ L2

        self.W2 -= (grad_W2 + lambda_l2 * self.W2) * learning_rate
        self.b2 -= np.sum(delta2, axis=0, keepdims=True) * learning_rate
        self.W1 -= (grad_W1 + lambda_l2 * self.W1) * learning_rate
        self.b1 -= np.sum(delta1, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs, learning_rate): 

        losses = []
        accuracies = []
        batch_size = 32

        for epoch in tqdm(range(epochs), desc='Treinando a MLP'):

            idx = np.random.permutation(X.shape[0])
            X_s = X[idx]
            y_s = y[idx]

            epoch_loss = 0
            correct = 0
            total = 0
            n_batches = 0

            for i in range(0, len(X_s), batch_size): 

                X_batch = X_s[i:i+batch_size]
                y_batch = y_s[i:i+batch_size]

                output = self.foward(X_batch)

                loss = -np.mean(
                    y_batch*np.log(output + 1e-8) +
                    (1-y_batch)*np.log(1-output + 1e-8)
                )

                epoch_loss += loss
                n_batches += 1

                pred = (output > 0.5).astype(int)
                correct += (pred == y_batch).sum()
                total += len(y_batch)

                self.backward(X_batch, y_batch, output, learning_rate)

            losses.append(epoch_loss / n_batches)
            accuracies.append(correct / total)

            if epoch % 20 == 0:
                tqdm.write(f'Época {epoch}, Loss={losses[-1]:.4f}, Acc={accuracies[-1]:.4f}')

        return losses, accuracies


    def predict(self, X):
        probs = self.foward(X)
        return (probs > 0.5).astype(int)