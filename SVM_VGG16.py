
# coding: utf-8

# In[14]:


from sklearn import svm
from LoadData import LoadData
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy import misc
import numpy as np


# In[2]:


# load data

train_X, test_X, train_Y, test_Y = LoadData().loadData('db1')


# In[3]:


clf = svm.SVC(kernel='linear', C=1).fit(train_X, train_Y)
clf.score(test_X, test_Y)


# Show Predict Example

# In[4]:


label_SVM = clf.predict(test_X)


# In[5]:


# get image list
imageData = LoadData().getImagePath('db1')


# In[28]:


count = test_Y.shape[0] - sum((label_SVM==test_Y)*1)
print(count)


# In[33]:


imageTest = imageData['testImages']
# create a fig to show image
fig = plt.figure(figsize=(20,10))

plt.title('Predict Wrong')
plt.axis('off')

predictTrue = label_SVM == test_Y
count = 0
# for all 0-9 labels
for i in range(test_Y.shape[0]):
    
    if not predictTrue[i]:
        # initialize subplots in a grid 2x5 at i+1th position
        image = misc.imread(imageTest[i], mode='RGB')
        ax = fig.add_subplot(2, 5, 1+count) 
        # display image
        ax.imshow(image, cmap=plt.cm.binary)
        ax.set_title("Predict: " + label_SVM[i] + "- True: " + test_Y[i])

        #don't show the axes
        plt.axis('off')
        count += 1
plt.show()


# Test Image
# 
