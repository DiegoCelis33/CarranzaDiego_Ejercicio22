#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[2]:


numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
print(np.shape(imagenes), n_imagenes) # Hay 1797 digitos representados en imagenes 8x8


# In[3]:


data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
print(np.shape(data))


# In[4]:


scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)


# In[5]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[6]:


loss = []
f1_score = []

loss_test = []
f1_score_test = []


neural = []

for i in range(20):
    
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(i+1), 
                                               max_iter=2000)
    mlp.fit(x_train, y_train)
    
    loss.append(mlp.loss_)
    f1_score.append(sklearn.metrics.f1_score(y_train, mlp.predict(x_train), average='macro'))
    
    loss_test.append(mlp.loss_)
    f1_score_test.append(sklearn.metrics.f1_score(y_test, mlp.predict(x_test), average='macro'))
    
    neural.append(i+1)
    


print('Loss', mlp.loss_)
print('F1', sklearn.metrics.f1_score(y_test, mlp.predict(x_test), average='macro'))


# In[7]:


loss = np.array(loss)
f1_score = np.array(f1_score)

plt.figure(figsize= (8,4))
plt.subplot(1,2,1)
plt.plot(neural,loss)
#plt.plot(neural,loss_test, label = "test")
plt.legend()
plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(neural,f1_score, label = "train")
plt.plot(neural,f1_score_test, label = "test")
plt.legend()
plt.title("F1 score")
plt.savefig("loss_f1.png")


# In[8]:


mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                               hidden_layer_sizes=(5), 
                                               max_iter=200)
mlp.fit(x_train, y_train)


# In[9]:


scale = np.max(mlp.coefs_[0])

plt.figure(figsize=(10, 5))

for i in range(5):
    l1_plot = plt.subplot(2, 5, i+1)
    l1_plot.imshow(mlp.coefs_[0][:,i].reshape(8, 8),cmap=plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('neurona %i' % i)


#run_time = time.time() - t0
#print('Example run in %.3f s' % run_time)
plt.savefig("neuronas.png")

