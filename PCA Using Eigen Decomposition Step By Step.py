#!/usr/bin/env python
# coding: utf-8

# # PCA Using Eigen Decomposition

# ### Import the required libraries

# In[280]:


import numpy as np
import math


# ### Create a matrix contains the following data

# In[281]:


f1=[1,5,1,5,8]
f2=[2,5,4,3,1]
f3=[3,6,2,2,2]
f4=[4,7,3,1,2]
all=np.array([f1,f2,f3,f4])
#print(all)
allvec=all.transpose()
#print(allvec)
#print(allvec[:,1])
#print(np.sum(allvec[:,1])/len(allvec[:,1]))
print(allvec.shape)


# ## Step 1: Standardize the dataset.

# ### Define a function to calculate data mean.
# 

# In[282]:


def meanfun (x):
    meanarr=[]
    for i in range (len(x[:,0])-1):
        y= np.sum(x[:,i])/len(x[:,i])
        meanarr.append(y)
    return meanarr
meanarray=np.array(meanfun(allvec))
print(meanarray)


# ### Check the function

# In[283]:


meanarray=np.array(meanfun(allvec))

feature1=meanarray[0]
feature2=meanarray[1]
print(meanarray)


# In[284]:


feature2=meanarray[1]

print(feature2)


# ### Define a function to calculate standard deviation of the data.
# 

# In[285]:


def stdfun (x,meanarr):
    stdarr=[]
    
 
    
    for i in range (len(x[:,0])-1):
        sum=0
        for j in range (len(x[0,:])+1):
              
            sum=sum+(pow(x[j,i]-meanarr[i],2))
              
              
        y=math.sqrt(sum/(len(x[:,0])-1))
        stdarr.append(y)
    return(stdarr)



# ### Check the function 

# In[286]:


stdarray=stdfun(allvec,meanarray)
print(stdarray)


# In[287]:


print(stdarray[1])


# ### Function to standardize the dataset.

# In[288]:


def standardfun (x,mean,std):
    standarray=np.zeros(x.shape)
    for i in range (len(x[:,0])-1) :
         for j in range (len(x[0,:])+1):
            standarray[j,i]= (x[j,i]-mean[i])/std[i]
              
    return standarray


# ### Apply The function to standardize the data

# In[289]:


stand_array=standardfun(allvec,meanarray,stdarray)
print(stand_array)


# ## Step 2: Calculate the covariance matrix for the features in the dataset.

# In[290]:


stand_tran=stand_array.transpose()
last_matrix=np.dot(stand_tran,stand_array)
cov_matrix=last_matrix/len(stand_array[:,0])


print(cov_matrix.shape)


# ## Step 3: Calculate the eigenvalues and eigenvectors for the covariance matrix.
# ## Step 4: Sort eigenvalues and their corresponding eigenvectors.

# In[291]:


val,vec=np.linalg.eig(cov_matrix)


# In[292]:


print(val)
print(vec)


# ## Step 5: Pick k eigenvalues and form a matrix of eigenvectors.

# ### Select the first eigen vectors

# In[293]:


print(val)
print(vec)
eigvectors=vec[0:2]
print(eigvectors.shape)


# ## Step 6:Transform the original matrix.

# In[294]:


last_features= np.dot(stand_array,eigvectors.transpose())
#print(stand_array)

print(last_features)


# ## Final  
# 

# In[295]:


print(last_features)

