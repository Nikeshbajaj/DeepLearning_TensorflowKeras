import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

class DeepNNtf(object):
    def __init__(self,inputunits=10, nclasses=3, Network=[10,5,3], alpha = 0.0001, batch_size = 32, showCostAt = 100):
        
        ops.reset_default_graph()
        
        self.nf = inputunits # number of features
        self.nC = nclasses
        #self.ny = 1 if nclasses==2 else nclasses
        self.ny = nclasses
        self.alpha = alpha
        self.batch_size = batch_size
        self.showCostAt = showCostAt
        
        self.Network = [self.nf] + Network + [self.ny]
                
        
        #self.X = tf.placeholder(tf.float32,[self.nf,None])
        #self.Y = tf.placeholder(tf.float32,[self.ny,None])
        
        self.X = tf.placeholder(tf.float32,[None,self.nf])
        self.Y = tf.placeholder(tf.float32,[None,self.ny])

        print("Network : ",self.Network)

        self.Para = self.initNetwork(self.Network)
        self.ZL   = self.fPropagation(self.X, self.Para)
        self.cost = self.compute_cost(self.ZL, self.Y)
        #self.pred = tf.equal(tf.argmax(self.ZL, 1), tf.argmax(self.Y, 1))
        #self.accuracy  = tf.reduce_mean(tf.cast(self.pred, tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.alpha).minimize(self.cost)
        
        self.costsAt =[]
        
        
    def fit(self,X,y,itr =10,batch_size=None):
        
        batch_size = self.batch_size if batch_size is None else batch_size

        self.init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            
            sess.run(self.init)
            
            for epoch in range(itr):
                
                epoch_cost = 0.
                nbatches = int(X.shape[0]/batch_size)
                
                XYbatches = self.create_batches(X,y,batch_size=batch_size)

                for batch in XYbatches:
                    (Xi, yi) = batch
                    _, icost = sess.run([self.optimizer, self.cost], feed_dict={self.X: Xi, self.Y: yi})
                    
                    epoch_cost += icost
                    
                epoch_cost /=nbatches
                self.costsAt.append(epoch_cost)
                
                # Print the cost every epoch
                if self.showCostAt>0 and epoch % self.showCostAt == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                    
    def initNetwork(self,Net =[2,7,3]):
        W = []
        b = []
        for i in range(len(Net)-1):
            Wi = tf.get_variable("W"+str(i+1), [Net[i+1],Net[i]], initializer = tf.contrib.layers.xavier_initializer())
            bi = tf.get_variable("b"+str(i+1), [Net[i+1],1], initializer = tf.zeros_initializer())
            W.append(Wi)
            b.append(bi)

        Para ={"W":W,"b":b}

        return Para
    
    def fPropagation(self,X,Para):
        A=[]
        A.append(tf.transpose(X))    
        W = Para["W"]
        b = Para["b"]
        
        L = len(W)
        for i in range(len(W)):
            Zi = tf.matmul(W[i],A[i]) + b[i]
            Ai_1 = tf.nn.relu(Zi)
            A.append(Ai_1)
        ZL = Zi
        return ZL
    def compute_cost(self,ZL, Y):
        log = tf.transpose(ZL)
        #y   = tf.transpose(Y)
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = log, labels = Y))
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = log, labels = Y))
        return cost
    
    def create_batches(self,X,y,batch_size):
        #assert X.shape[1] == y.shape[1]
        assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        Idx = np.random.permutation(n)
        batches = []
        for i in range(n//batch_size):
            Idxi = Idx[i*batch_size:(i+1)*batch_size]
            #Xi = X[:,Idxi]
            #yi = y[:,Idxi]
            Xi = X[Idxi,:]
            yi = y[Idxi,:]
            batches.append([Xi,yi])
            
        if (i+1)*batch_size !=len(Idx):
            Idxi = Idx[(i+1)*batch_size:]
            Xi = X[Idxi,:]
            yi = y[Idxi,:]
            batches.append([Xi,yi])
        return batches
    
    def plotCost(self,ax =None):
        ax =plt.gca() if ax is None else ax
        epochs = range(1,len(self.costsAt)+1)
        ax.plot(epochs,self.costsAt)
        ax.set_xlim([epochs[0],epochs[-1]])
        ax.grid()
        ax.set_xlabel('Epocs')
        ax.set_ylabel('Cost')