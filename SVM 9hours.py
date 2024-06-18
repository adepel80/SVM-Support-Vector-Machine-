#!/usr/bin/env python
# coding: utf-8

# # USING SVM FOR CLASSIFICATION

# In[5]:


import numpy as np
import pandas as pd

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


pwd


# In[8]:


cupmuff = pd.read_csv('cupmuff.csv')


# In[10]:


cupmuff.head(4)


# In[15]:


#plot our data
sns.lmplot('Flour','Sugar', data=cupmuff, hue='Type',
          palette='Set1',fit_reg=False, scatter_kws={'s': 70});


# In[17]:


# format or pre process our data 
type_label = np.where(cupmuff['Type']=='Muffin', 0,1)
cupmuff_features = cupmuff.columns.values[1:].tolist()
cupmuff_features
ingredients = cupmuff[['Flour', 'Sugar']].values
ingredients


# In[18]:


#fit model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)


# In[20]:


# GET THE sepdarting hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx =np.linspace(30,60)
yy = a* xx - (model.intercept_[0]) / w[1]

#plot d parallel to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a*b[0])
b = model.support_vectors_[-1]
yy_up = a * xx +(b[1] - a*b[0])


# In[23]:


sns.lmplot('Flour','Sugar', data=cupmuff, hue='Type', palette='Set1', fit_reg=False, scatter_kws={'s': 70})
plt.plot(xx,yy,linewidth = 2, color ='black')
plt.plot(xx, yy_down,'k--')
plt.plot(xx, yy_up, 'k--')


# # predict if 50 parts flour and 20 parts sugar

# In[30]:


# create a function to create muffin or cupcake
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('you are looking at a muffin recipe')
    else:
        print('you are looking at cupcake recipee')
# predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50,20)


# In[27]:


# let  plot this on graph
sns.lmplot('Flour','Sugar', data=cupmuff, hue='Type', palette='Set1', fit_reg=False,scatter_kws={'s': 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50,20, 'yo', markersize ='9')


# In[31]:


# PREDICT if 40 parts flour and 20 parts flour
muffin_or_cupcake(40,20)


# In[ ]:




