#!/usr/bin/env python
# coding: utf-8

# In[24]:



# import random
# import numpy as np

# # In[25]:


# def reassign(data_index_x,data_index_y,cluster_index,u,n_clusters) :
#     for i in range(n_clusters) :
#         if i==cluster_index :
#             u[data_index_x][data_index_y][i]=1
#         else :
#             u[data_index_x][data_index_y][i]=0


# # In[26]:


# def init_membership_random(n_points_x,n_points_y,n_clusters):
#     u = np.zeros((n_points_x,n_points_y, n_clusters))
#     for i in range(n_points_x):
#         for j in range(n_points_y):
#             row_sum = 0.0
#             for c in range(n_clusters):
#                 if c == n_clusters-1: 
#                     u[i][j][c] = 1.0 - row_sum
#                 else:
#                     rand_clus = random.randint(0, n_clusters-1)
#                     rand_num = random.random()
#                     rand_num = round(rand_num, 2)
#                     if rand_num + row_sum <= 1.0:
#                         u[i][j][c] = rand_num
#                         row_sum = row_sum +rand_num
#     return u            
              


# # In[27]:


# def distance_squared( x, c):
#         sum_of_sq = 0.0
#         for i in range(len(x)):
#             sum_of_sq = sum_of_sq +(x[i]-c[i]) ** 2
#         return sum_of_sq


# # In[28]:


# def compute_cluster_centers(image,u,n_clusters,m):
#         n_points_x,n_points_y = image.shape
        
#         new_image=np.zeros(u.shape)
#         for c in range(n_clusters) :
#             new_image[:,:,c]=image
        
#         sum_1=np.sum(np.sum((u**m)*new_image,axis=0),axis=0)
#         sum_2=np.sum(np.sum((u**m),axis=0),axis=0)
    
#         centers = sum_1/sum_2       
#         return centers     


# # In[29]:


# '''def update_u(data,center,u,m) :
#     n_points_x,n_points_y = data.shape
#     n_clusters=len(center)
    
#     distance=np.zeros(u.shape)
#     for c in range(n_clusters) :
#         distance[:,:,c]=(data-center[c])**2
        
#     for i in range(n_points_x):
#         for j in range(n_points_y):
#             dist = distance[i,j,:]
#             update=True 
#             for c in range(n_clusters):
#                 distijc=dist[c]
#                 if distijc==0 :
#                     reassign(i,j,c,u,n_clusters)
#                     update = False
#                     break
#             if update==True :    
#                 for c in range(n_clusters) :
#                     sum_1=0
#                     d1=dist[c]
#                     for k in range(n_clusters) :
#                         sum_1=sum_1+((d1/dist[k])**(1/(m-1)))
#                     u[i][j][c]=(1/sum_1)'''  
# def update_u(data,center,u,m) :
#     n_points_x,n_points_y = data.shape
#     n_clusters=len(center)
#     u = np.zeros((n_points_x,n_points_y,n_clusters))
#     u_ = np.zeros((n_points_x,n_points_y,n_clusters))

#     distance=np.zeros(u.shape)
#     for c in range(n_clusters) :
#         distance[:,:,c]=(data-center[c])**2
#     u[distance==0]=2
#     u_sum = np.sum(u,axis=2)
#     u_sum=u_sum/2
#     for c in range(n_clusters) :
#       u[:,:,c]=u[:,:,c]-u_sum
#     pos = u==1  
#     neg = u==-1
#     distance[distance==0] =0.1
#     dist_inv = (1/distance)**(1/(m-1))
#     dist_inv_sum = np.sum(dist_inv, axis=2)
#     for c in range(n_clusters):
#       u_[:,:,c] = dist_inv_sum/dist_inv[:,:,c] 
#     u_[pos] =1
#     u_[neg]=0

#     return u_
     

# # In[30]:


# def fit(X,n_clusters,max_iter,m, error):
#         X = np.array(X)
#         n_points_x,n_points_y = X.shape
#         u=init_membership_random(n_points_x,n_points_y,n_clusters)
#         centers = compute_cluster_centers(X,u,n_clusters,m)
#         for i in range(max_iter):
#             prev_centers=centers
#             u = update_u(X,centers,u,m)
#             centers = compute_cluster_centers(X,u,n_clusters,m)
#             print(centers)
#             #if np.max(abs(np.array(prev_centers)-np.array(centers))) <= error:
#                 #return u,centers
#         return u, centers


# # In[ ]:



import random
import numpy as np

# In[25]:


def reassign(data_index_x,data_index_y,cluster_index,u,n_clusters) :
    for i in range(n_clusters) :
        if i==cluster_index :
            u[data_index_x][data_index_y][i]=1
        else :
            u[data_index_x][data_index_y][i]=0


# In[26]:


def init_membership_random(n_points_x,n_points_y,n_clusters):
    u = np.zeros((n_points_x,n_points_y, n_clusters))
    for i in range(n_points_x):
        for j in range(n_points_y):
            row_sum = 0.0
            for c in range(n_clusters):
                if c == n_clusters-1: 
                    u[i][j][c] = 1.0 - row_sum
                else:
                    rand_clus = random.randint(0, n_clusters-1)
                    rand_num = random.random()
                    rand_num = round(rand_num, 2)
                    if rand_num + row_sum <= 1.0:
                        u[i][j][c] = rand_num
                        row_sum = row_sum +rand_num
    return u            
              


# In[27]:


def distance_squared( x, c):
        sum_of_sq = 0.0
        for i in range(len(x)):
            sum_of_sq = sum_of_sq +(x[i]-c[i]) ** 2
        return sum_of_sq


# In[28]:


def compute_cluster_centers(image,u,n_clusters,m):
        n_points_x,n_points_y = image.shape
        
        new_image=np.zeros(u.shape)
        for c in range(n_clusters) :
            new_image[:,:,c]=image
        
        sum_1=np.sum(np.sum((u**m)*new_image,axis=0),axis=0)
        sum_2=np.sum(np.sum((u**m),axis=0),axis=0)
    
        centers = sum_1/sum_2       
        return centers     


# In[29]:


def update_u(data,center,u,m) :
    n_points_x,n_points_y = data.shape
    n_clusters=len(center)
    
    distance=np.zeros(u.shape)
    for c in range(n_clusters) :
        distance[:,:,c]=(data-center[c])**2
        
    for i in range(n_points_x):
        for j in range(n_points_y):
            dist = distance[i,j,:]
            update=True 
            for c in range(n_clusters):
                distijc=dist[c]
                if distijc==0 :
                    reassign(i,j,c,u,n_clusters)
                    update = False
                    break
            if update==True :    
                for c in range(n_clusters) :
                    sum_1=0
                    d1=dist[c]
                    for k in range(n_clusters) :
                        sum_1=sum_1+((d1/dist[k])**(1/(m-1)))
                    u[i][j][c]=(1/sum_1)  


# In[30]:


def fit(X,n_clusters,max_iter,m, error):
        X = np.array(X)
        n_points_x,n_points_y = X.shape
        u=init_membership_random(n_points_x,n_points_y,n_clusters)
        centers = compute_cluster_centers(X,u,n_clusters,m)
        for i in range(max_iter):
            prev_centers=centers
            update_u(X,centers,u,m)
            centers = compute_cluster_centers(X,u,n_clusters,m)
            if abs(np.max(np.array(prev_centers)-np.array(centers)))<=error :
                return u, centers
        return u, centers
