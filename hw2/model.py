import sys
import numpy as np

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return res
    
class generative_model:
    def __init__(self):
        self.w = 0
        self.b = 0
        
    def train(self,train_data,train_labels):
        people = train_data.shape[0]
        num_w = train_data.shape[1] # number of weight
        
    ########## Classify Train_data ##########
        train_0 = []
        train_1 = []
        for i in range(people):
            if train_labels[i] == 0:
                train_0.append(train_data[i])
            else:
                train_1.append(train_data[i])

        train_0 = np.array(train_0)
        train_1 = np.array(train_1)        
    ########## Calculate mu & sigma ##########
        num_0 = train_0.shape[0]
        num_1 = train_1.shape[0]
        
        mu_0 = np.mean(train_0,axis=0)
        mu_1 = np.mean(train_1,axis=0)
        
        sigma_0 = np.zeros((num_w,num_w))
        sigma_1 = np.zeros((num_w,num_w))
        
        for i in range(num_0):
            sigma_0 += np.dot(np.transpose([train_0[i]-mu_0]), [train_0[i]-mu_0] )
        for i in range(num_1):
            sigma_1 += np.dot(np.transpose([train_1[i]-mu_1]), [train_1[i]-mu_1] )

        sigma_0 = sigma_0 / num_0
        sigma_1 = sigma_1 / num_1
        sigma = (num_0*sigma_0 + num_1*sigma_1)/people
        
    ########## Calculate w & b ###############
        sigma_inv = np.linalg.inv(sigma)
        self.w = np.dot( (mu_1-mu_0), sigma_inv )
        self.b = np.log(float(num_1) / num_0) - 0.5 * np.dot(np.dot(mu_1,sigma_inv),mu_1) \
                             + 0.5 * np.dot(np.dot(mu_0,sigma_inv),mu_0)

        return self.w, self.b
    
    def test(self,test_data):
        output = []        
        z = np.dot(self.w, test_data.T) + self.b 
        for i in range(z.shape[0]):
            if z[i] > 0 :
                output.append(1)
            else:
                output.append(0)
        return output
        
class logistic_regression:
    def __init__(self, batch_size, epoch, lr_w, lr_b, lamda):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr_w = lr_w
        self.lr_b = lr_b
        self.lamda = lamda
        self.w = 0
        self.b = 0
    
    def train(self,train_data,train_labels):
        self.b  = 0
        self.w = np.zeros(train_data.shape[1]).reshape(train_data.shape[1],1)
        
        previous_bias = 1
        previous_weight = np.ones(train_data.shape[1]).reshape(train_data.shape[1],1)
        
        error_hisory = []
        
        for i in range(self.epoch): 
            error = 0
            for j in range(int(train_data.shape[0]/self.batch_size)):
                x = train_data[j*self.batch_size:j*self.batch_size+self.batch_size,:]            
                y_predict = sigmoid(self.b  + np.dot(x,self.w))
                y = train_labels[j*self.batch_size:j*self.batch_size+self.batch_size].reshape(self.batch_size,1)
                
                weight_gradient = (-np.dot(np.transpose(x),(y-y_predict)) + self.lamda*self.w) / self.batch_size
                bias_gradient = -2*(np.sum((y-y_predict))) / self.batch_size
                
                # adagrad
                previous_weight = previous_weight + weight_gradient**2
                previous_bias =  previous_bias + bias_gradient**2            
                self.w = self.w - self.lr_w/np.sqrt(previous_weight) * weight_gradient
                self.b  = self.b  - self.lr_b/np.sqrt(previous_bias) * bias_gradient      
                
                error = error + (y-y_predict)**2
                
            error_hisory.append(np.sqrt(np.sum(error)/j/self.batch_size))
            
        return self.w, self.b , error_hisory
        
    def test(self,test_data):
        output = []        
        z = np.dot(test_data, self.w) + self.b 
        for i in range(z.shape[0]):
            if z[i,0] > 0 :
                output.append(1)
            else:
                output.append(0)
        return output


