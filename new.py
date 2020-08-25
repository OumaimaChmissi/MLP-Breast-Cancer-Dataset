import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
#----------------------------------------------------------
#1)changer le vecteur label (g,b) to (1,0)

def change_to_numeric(x):
    if x=='B':
        return 1
    if x=='M':
        return 0


#--------------------------------------------------------------------
#2) importation de la base de donnees
dataset = pd.read_csv("data.csv")
dataset['diagnosis']= dataset['diagnosis'].apply(change_to_numeric)

y = dataset['diagnosis']
Data = dataset.drop(["diagnosis"], axis=1)
Data = dataset.drop(["id"], axis=1)
Data = dataset.drop(["Unnamed: 32"], axis=1)
X=np.array(Data)
print("Affichage du vecteur X")

print(X)
print("/////////////////////////////////////")
print("Affichage du vecteur y")
print(y)
#----------------------------------------------------------------------------------------
#3) Declaration
num_i_units = 32 # nb des entrees
num_h_units = 10  # nb de hidden layer 1
# num_h_units2 = 8 #nb de hidden layer 2
num_o_units = 1  #nb de output


learning_rate = 0.01 # 0.001, 0.01 <- Magic values
reg_param = 0 # 0.001, 0.01 <- Magic values
max_iter =10000 # 5000 <- Magic value
m = 400 # Number of training examples

# The model needs to be over fit to make predictions. Which
np.random.seed(2)
W1 = np.random.normal(0, 1, (num_h_units, num_i_units)) # 2x32
W2 = np.random.normal(0, 1, (num_o_units, num_h_units)) # 1x2
# W3 = np.random.normal(0, 1, (num_o_units, num_h_units2)) # 1x2

B1 = np.random.random((num_h_units, 1)) # 2x1
# B12 = np.random.random((num_h_units2, 1)) # 2x1

B2 = np.random.random((num_o_units, 1)) # 1x1

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))     


def tanh(t, derv=False):
    if derv: return 1-t**2
    return (np.exp(t)-np.exp(-t))/(np.exp(t)+np.exp(-t))


def forward(x, predict=False):
    a1 = x.reshape(x.shape[0], 1) # Getting the training example as a column vector.

    z2 = W1.dot(a1) + B1 # 2x2 * 2x1 + 2x1 = 2x1
    a2 = sigmoid(z2) # 2x1

    # z21 = W3.dot(a2) + B12  # 2x2 * 2x1 + 2x1 = 2x1
    # a21 = tanh(z21)  # 2x1

    z3 = W2.dot(a2) + B2 # 1x2 * 2x1 + 1x1 = 1x1
    a3 = tanh(z3)

    if predict: return a3
    return (a1, a2, a3)

dW1 = 0
dW2 = 0
# dW3= 0

dB1 = 0
# dB12 = 0
dB2 = 0

cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    c = 0

    dW1 = 0
    dW2 = 0
    # dW3 = 0

    dB1 = 0
    # dB12 = 0
    dB2 = 0

    for j in range(m):
        sys.stdout.write("\rIteration: {} and {}".format(i + 1, j + 1))

        # Forward Prop.
        a0 = X[j].reshape(X[j].shape[0], 1) # 2x1

        z1 = W1.dot(a0) + B1 # 2x2 * 2x1 + 2x1 = 2x1
        a1 = sigmoid(z1) # 2x1

        # z12 = W3.dot(a1) + B12  # 2x2 * 2x1 + 2x1 = 2x1
        # a12 = tanh(z12)

        z2 = W2.dot(a1) + B2 # 1x2 * 2x1 + 1x1 = 1x1
        a2 = tanh(z2) # 1x1


        # Back prop.
        dz2 = a2 - y[j] # 1x1
        dW2 += dz2 * a1.T # 1x1 .* 1x2 = 1x2

        dz1 = np.multiply((W2.T * dz2), tanh(a1,derv=True)) # (2x1 * 1x1) .* 2x1 = 2x1
        dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2

        dB1 += dz1 # 2x1
        dB2 += dz2 # 1x1

        c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
        sys.stdout.flush() # Updating the text.
    W1 = W1 - learning_rate * (dW1 / m) + ( (reg_param / m) * W1)
    W2 = W2 - learning_rate * (dW2 / m) + ( (reg_param / m) * W2)
    # W3 = W3 - learning_rate * (dW3 / m) + ( (reg_param / m) * W3)

    B1 = B1 - learning_rate * (dB1 / m)
    # B12 = B12 - learning_rate * (dB12 / m)
    B2 = B2 - learning_rate * (dB2 / m)
    cost[i] = (c / m) + (
        (reg_param / (2 * m)) *
        (
            np.sum(np.power(W1, 2)) +
            np.sum(np.power(W2, 2)) 
            # np.sum(np.power(W3, 2))

        )
    )


for x in X:
    print("\n")
    print(x)
    print(forward(x, predict=True))

plt.plot(range(max_iter), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()