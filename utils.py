#!/usr/bin/env python
# coding: utf-8

# In[70]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
from csv import writer


# In[71]:


def fuzzy_to_crisp(u) :
    x1 , y1 , c = u.shape
    im=np.zeros((y1))
    for i in range(x1) :
        for j in range(y1):
            l=max(u[i,j,:])
            for k in range(c) :
                if u[i,j,k]==l :
                    u[i,j,k]=1
                else :
                    u[i,j,k]=0
    return u


# In[72]:


def clustering(u) :
    x1,y1,c=u.shape
    cluster_set = []
    for i in range(c):
        temp = u[:,:,i]
        cluster_set.append(temp)
    return cluster_set 


# In[73]:


def indexing_new(set_1 , set_2 , A) :
    cluster_set=[]
    intensity_set = []
    
    for set_j in set_1 :
        intense = np.sum(np.multiply(set_j,A))/np.sum(set_j)
        cluster_set.append(set_j)
        intensity_set.append(intense)
        
    new_cluster_set=sorting(intensity_set , cluster_set)
    proper_set = []
    for i in range(len(new_cluster_set)) :
        proper_set.append([set_2[i][1] , new_cluster_set[i]])
    return proper_set


# In[74]:


def indexing_final(clustered_set , set_2 , A ):
    avg_intensity = list()
    sorted_avg_intensity = list()
    sorted_clustered_set = list()
    
    for set_j in clustered_set:
        value = np.sum( np.multiply( set_j , A ) )/np.sum(set_j)
        avg_intensity.append(value)
    
    #for j, set_j in enumerate(clustered_set):
        #clustered_images[j] = set_j
        
    sorted_avg_intensity =sorted(avg_intensity)
    
    for i in range(len(sorted_avg_intensity)):
        for set_j in clustered_set:
            if (np.sum(np.multiply(set_j,A))/np.sum(set_j)) == sorted_avg_intensity[i]: 
                sorted_clustered_set.append( ( set_2[i][1] , set_j ) )
    return sorted_clustered_set


# In[75]:


def sorting(set_1 , set_2) :
    intense_unsort_set = set_1
    set_1.sort()
    intense_sort_set = set_1
    cluster_set=[]
    for i in range(len(set_1)) :
            j = intense_unsort_set.index(intense_sort_set[i])
            cluster_set.append(set_2[j])
    return cluster_set


# In[76]:


def intersect(A,B) :
    x1, y1 = A.shape    
    C=np.zeros((x1,y1))
    for i in range(x1) :
        for j in range(y1) :
            if A[i][j]==1 and B[i][j]==1 :
                    C[i][j]=1
    return C  


# In[77]:


def dice_score(clustered,ground) :
    intersection=intersect(clustered,ground)
    d1=2*(np.sum(intersection))
    d2=(np.sum(clustered))+(np.sum(ground))
    return d1/d2


# In[78]:


def jaccard_score(clustered,ground) :
    intersection = intersect(clustered,ground)
    d1=np.sum(intersection)
    d2=(np.sum(clustered))+(np.sum(ground))-(np.sum(intersection))
    return d1/d2

def FPR_score(clustered,ground) :
    intersection = intersect(clustered,ground)
    return (np.sum(clustered) - np.sum(intersection))/np.sum(ground)

def FNR_score(clustered, ground) :
    intersection = intersect(clustered,ground)
    return (np.sum(ground) - np.sum(intersection))/np.sum(ground)
    
# In[79]:


def tot_jaccard_score(pro_set) :
    all_d=[]
    for set_i in pro_set :
        all_d.append(jaccard_score(set_i[0], set_i[1]))
    return all_d 


# In[80]:


def tot_dice_score(pro_set) :
    all_d=[]
    for set_i in pro_set :
        all_d.append(dice_score(set_i[0], set_i[1]))
    return all_d

def tot_FPR_score(pro_set) :
    all_fpr = []
    for set_i in pro_set :
        all_fpr.append(FPR_score(set_i[0], set_i[1]))
    return all_fpr 

def tot_FNR_score(pro_set) :
    all_fnr = []
    for set_i in pro_set :
        all_fnr.append(FNR_score(set_i[0], set_i[1]))
    return all_fnr 

def PC(u,m) :
    return np.sum(u**m)/(u.shape[0]*u.shape[1])

def PE(u) :
    u_ = u
    u_[u==0] = 1
    u_ = np.log(u_)
    return np.sum(u_*u)/(u.shape[0]*u.shape[1])*(-1)
# In[81]:


def reshaping_linear(image):
    x,y=image.shape
    image2d=image.reshape(x*y,1)
    return image2d


# In[82]:


def grounded_imaging(image):
    image_act = np.zeros(image.shape)
    unique_val = np.unique(image)
    
    image1=np.zeros((420,456))
    image2=np.zeros((420,456))
    image3=np.zeros((420,456))
    image4=np.zeros((420,456))
    for i in range(420):
        for j in range(456):
            if image[i,j]==4:
                image1[i,j]=1
            if image[i,j]==76:
                image2[i,j]=1
            if image[i,j]==148:
                image3[i,j]=1
            if image[i,j]==201:
                image4[i,j]=1            
            
    for i in range(420):
        for j in range(456):
            if image1[i][j] == 1:
                image_act[i][j] = 0
            if image2[i][j] == 1:
                image_act[i][j] = 1
            if image3[i][j] == 1:
                image_act[i][j] = 2
            if image4[i][j] == 1:
                image_act[i][j] = 3
    
    ground_set=[(4,image1),(76,image2),(148,image3),(201,image4)]
    return ground_set , image_act


# In[83]:


dice_score_list = list()
jaccard_score_list = list()
accuracy_list = list()
FPR_list = list()
FNR_list = list()
PC_list = list()
PE_list = list()


# In[84]:



def processing(u , m, image , ground_set , image_act, image_name):
    
    #JACCARD AND DICE SCORE
    x,y = image.shape
    
    PC_ = PC(u,m)
    PE_ = PE(u)
    u = fuzzy_to_crisp(u)
    clustered_set = clustering(u)
    proper_set = indexing_final(clustered_set,ground_set,image)
    dice_score=tot_dice_score(proper_set)
    jaccard_score=tot_jaccard_score(proper_set)
    FPR = tot_FPR_score(proper_set)
    FNR = tot_FNR_score(proper_set)
    
    image_pred=np.zeros((420,456))
    for i in range(420) :
        for j in range(456) :
            if proper_set[0][1][i][j] == 1 :
                image_pred[i][j]=0
            if proper_set[1][1][i][j] == 1 :
                image_pred[i][j]=1  
            if proper_set[2][1][i][j] == 1 :
                image_pred[i][j]=2
            if proper_set[3][1][i][j] == 1 :
                image_pred[i][j]=3
    
    #ACCURACY
    
    sum_1 = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image_pred[i][j] == image_act[i][j]:
                sum_1 = sum_1 + 1
            else:
                continue
            
    accuracy = (sum_1/(image.shape[0]*image.shape[1]))*100 
    accuracy_list.append(accuracy)
                
    image_act = image_act.reshape(x*y,1)
    image_pred = image_pred.reshape(x*y,1)
    
    #CONFUSION MATRIX
    
    conf_mat = confusion_matrix(image_act, image_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                xticklabels=['Pent','Cir','Hex','Bg'], 
                yticklabels=['Pent','Cir','Hex','Bg'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(image_name, size=16)
    

    
    return dice_score , jaccard_score , accuracy, FPR, FNR, PC_ , PE_


# In[86]:


def append_result(file_name,avg_score,Method):
    new_row=[Method]
    for i in avg_score:
        new_row.append(i)
    with open(file_name, 'a+', newline='') as write_obj :
        csv_writer=writer(write_obj)
        csv_writer.writerow(new_row)        


# In[37]:
