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

#Definindo a função de ativação sigmoid e seu gradiente 

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def grad_sigmoid(x): 
    return x * (1 - x)

#Arquitetura da Rede MLP 

class MLP: 
    def __init__(self, input_size, hidden_size, output_size): 

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def foward(self, X): 

        self.z1 = np.dot(X, self.W1) + self.b1 
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2 
        self.a2 = sigmoid(self.z2)

        return self.a2 
        
    def backward(self, X, y, output, learning_rate):

        delta2 = output - y

        delta1 = delta2.dot(self.W2.T) * grad_sigmoid(self.a1)

        self.W2 -= self.a1.T.dot(delta2) * learning_rate
        self.b2 -= np.sum(delta2, axis=0, keepdims=True) * learning_rate
        self.W1 -= X.T.dot(delta1) * learning_rate
        self.b1 -= np.sum(delta1, axis=0, keepdims=True) * learning_rate
        
    def train(self, X, y, epochs, learning_rate): 

        losses = []

        accuracies = []

        for epoch in range(epochs): 
            output = self.foward(X)

            loss = -np.mean(
                        y*np.log(output + 1e-8) +
                        (1-y)*np.log(1-output + 1e-8)
                    )
                
            losses.append(loss)

            pred = (output > 0.5).astype(int)
            acc = (pred == y).mean()
            accuracies.append(acc)

            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0: 
                print(f'Época {epoch}, Perda = {loss}')

        return losses, accuracies  


    def predict(self, X):
        probs = self.foward(X)
        return (probs > 0.5).astype(int)