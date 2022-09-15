from global_parameters import parameters as p
import numpy as np
import time

# Specific for mlp
class MLP:
    def __init__(self, rate, shape):
        self.rate = rate
        #self.nin = nin
        #self.nhidden = nhidden
        #self.nout = nout
        self.shape = shape

        self.init_wb()
        
    #
    def init_wb(self):
        
        win = np.random.random_sample((self.shape[1], self.shape[0])) * (1/255)
        wh = np.random.random_sample((self.shape[2], self.shape[1])) * (1/255)
        
        bias_in = np.random.random_sample(self.shape[1]) * (1/255)
        bias_h = np.random.random_sample(self.shape[2]) * (1/255)

        self.w = [win, wh]
        self.bias = [bias_in, bias_h]
    
    def activation(self, x):
        return 1/(1 + np.exp(-x))
    
    def feedforward(self, raw):
        self.inputL = raw        
        self.hiddenL = self.activation(np.matmul(self.inputL, self.w[0].T) + self.bias[0])
        self.outputL = self.activation(np.matmul(self.hiddenL, self.w[1].T) + self.bias[1])
        #print('Hidden shape: ', self.hiddenL.shape)
        #print('Output shape: ', self.outputL.shape)
    
    def backpropagation_output(self, y):
        yhat = self.outputL
        x1 = self.hiddenL
        delta = (yhat * (y - yhat) * (1-yhat))

        self.w[1] = self.w[1] + 2 * self.rate * np.matmul(delta.reshape((self.shape[2], 1)), x1.reshape((1, self.shape[1])))
        self.bias[1] = self.bias[1] + 2 * self.rate * (delta)
    
    def backpropagation_hidden(self, y):
        yhat = self.outputL
        x0 = self.inputL
        x1 = self.hiddenL
        delta1 = (yhat * (y - yhat) * (1-yhat))
        delta0 = (x1 * (1-x1))
        #print('delta0', delta0)
        #print('delta1', delta1)
        
        
        self.w[0] = self.w[0] + 2 * self.rate * np.matmul( ((delta0 * np.matmul(delta1, self.w[1])).reshape(self.shape[1], 1)), x0.reshape(1, self.shape[0]))
        self.bias[0] = self.bias[0] + 2 * self.rate * (delta0 * np.matmul(delta1, self.w[1]))

    def training(self, totrain, totrain_labels, step_check = p['step_check'], path = 'output/Loss-output.txt'):
        fd = open(path, 'w')
        
        for i in range(len(totrain)):
            # Print loss based on step
            if (i%step_check == 0 and i > 0):
                print('Loss[{0}] = {1}'.format(i, loss))

            # Formatting label
            y = np.zeros(10)
            y[totrain_labels[i]] = 1

            # Training
            self.feedforward(totrain[i]/255)
            loss = self.loss(y)
            self.backpropagation_output(y)
            self.backpropagation_hidden(y)
            fd.write(str(loss) + '\n')

        fd.close()
        
    def test(self, totest, totest_labels, path = 'output/predicted.txt'):
        hit = 0
        fd = open(path, 'w')

        fd.write('[predicted, class]\n')
        for i, test in enumerate(totest):
            # Predict
            self.feedforward(test/255)

            # Getting predicted
            predicted = np.argmax(self.outputL)
            cls = totest_labels[i]

            # Compare
            if predicted == cls:
                hit += 1

            fd.write('[{0}, {1}]\n'.format(predicted, cls))

        fd.close()
        
        acc = hit/len(totest)
        return acc, 1-acc
            
    
    def loss(self, target):
        return ((target - self.outputL)**2).sum()
