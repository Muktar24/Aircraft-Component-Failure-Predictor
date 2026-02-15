import numpy as np

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1/(1+np.exp(-z))
def initialization(n_input,n_hidden,n_output=1):
    W1 = np.random.rand(n_input + 1,n_hidden) * 0.001 # 9 * 8
    W2 = np.random.rand(n_hidden + 1 ,n_output) * 0.001  # 9*1
    return W1,W2

def forward_propagation(X,W1,W2):
    X_= np.c_[np.ones((X.shape[0], 1)),X] #16000*9
    Z1=X_@W1 #16000*8
    A1=sigmoid(Z1)#16000*8
    A1_= np.c_[np.ones((A1.shape[0], 1)),A1]#16000*9
    Z2=A1_@W2#16000*1
    A2=sigmoid(Z2)#16000*1
    return X_,Z1,A1,A1_,Z2,A2
#

def backward_propagation(Y,A1,A1_,W2,A2,X_):
    Y = Y.reshape(-1, 1) #16000*1
    m=Y.shape[0] #16000
    dz2=A2-Y #16000*1
    dw2=1/m*(A1_.T@dz2) # 9*1
    W2_=W2[1:,:]
    dz1=dz2@W2_.T*(A1*(1-A1))#16000*8*16000*8
    dw1=1/m*(X_.T@dz1)#9*8

    return dw1,dw2


def train(X,Y,iter=1000,lr=0.1):

    W1,W2=initialization(X.shape[1],8,1)

    for i in range(iter):
        X_,Z1,A1,A1_,Z2,A2 = forward_propagation(X, W1, W2)
        dw1,dw2=backward_propagation(Y,A1,A1_,W2,A2,X_)
        W1-=lr*dw1
        W2-=lr*dw2
    return W1,W2

def predict(X,W1,W2,thres=0.5):
    _, _, _, _, _, A2 = forward_propagation(X, W1, W2)
    return A2
