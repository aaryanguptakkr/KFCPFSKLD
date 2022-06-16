#!/usr/bin/env python
# coding: utf-8

# In[79]:


import random
import numpy as np


# In[80]:


def k_fun(x,y,sigma) :
    d1=(sigma**2)*(-1)
    d2=(x-y)**2/d1
    return np.exp(d2)


# In[81]:


def u_norm(u,n,e,n_clusters) :
    n_points_x, n_points_y, n_clusters = u.shape
    new_u = np.zeros(u.shape)
    for i in range(n_points_x) :
        for j in range(n_points_y) :
            sum_=0.0
            for c in range(n_clusters) :
                if 1-n[i,j,c]-e[i,j,c] == 0 :
                    sum_ = sum_ + 0
                else :
                    sum_ = sum_ + u[i,j,c]/(1-n[i,j,c]-e[i,j,c])
            for c in range(n_clusters) :
                new_u[i,j,c] = u[i,j,c]/sum_ 
    return new_u     


# In[82]:


def reassign(data_index_x,data_index_y,cluster_index,u,n_clusters) :
    for i in range(n_clusters) :
        if i==cluster_index :
            u[data_index_x][data_index_y][i]=1
        else :
            u[data_index_x][data_index_y][i]=0


# In[98]:

# In[99]:


def cal_centers(X,u,n,m,e,n_clusters,win_size,prev_centers,sigma):
        centers = []
        u=u*(2-e)
        for c in range(n_clusters):
            centers.append(np.sum(u[:,:,c]*X*k_fun(X,prev_centers[c],sigma))/np.sum(u[:,:,c]* k_fun(X,prev_centers[c],sigma)))
        return centers    


# In[100]:

# In[101]:


def update_u(data,u,n_clusters,m,center,n,e,gamma,win_size,sigma ) :
    n_points_x,n_points_y = data.shape
    pi=cal_spatial_wt(u,n,e,win_size)
    dist = np.zeros((n_points_x,n_points_y,n_clusters))
    for c in range(n_clusters) :
        dist[:,:,c] = pi[:,:,c] /np.exp((1-k_fun(data,center[c],sigma))/gamma)
    dist_sum = np.sum(dist, axis=2)
    for c in range(n_clusters) :
        u[:,:,c] = (dist[:,:,c]/dist_sum)/(2-e[:,:,c])
    return u    


# In[102]:


def update_n(n, e, n_clusters) :
    n_e=n+e
    n_e[n_e==0]=1
    new_n=(e*(n_e+e))/(e+n_e)
    return new_n/n_clusters 


# In[103]:



def update_e(u, n, alpha) :
    u_n=u+n
    u_n[u_n>1]=1
    e = 1-(u_n)-(1-(u_n)**alpha)**(1/alpha)           
    return e


# In[104]:


def cal_spatial_wt(u,n,e,win_size) :
    n_points_x, n_points_y, n_clusters = u.shape
    u_spat = np.zeros((n_points_x, n_points_y, n_clusters))
    u=u*(2-e)
    zero = n_clusters*[0]
    u=np.insert(u,n_points_x,zero,axis=0)
    u=np.insert(u,0,zero,axis=0)
    u=np.insert(u,n_points_y,zero,axis=1)
    u=np.insert(u,0,zero,axis=1)
    for i in range(win_size) :
        for j in range(win_size) :
            if i==win_size//2 and j==win_size//2 :
                continue
            else :    
                temp=u[i:n_points_x+i , :]
                temp=temp[: , j:n_points_y+j]
                u_spat=u_spat + temp       
    return u_spat/(win_size*win_size-1)   


# In[105]:


def fit_(image, n_clusters, m, alpha, error, iterations, init_u,init_cent,gamma,win_size,sigma) :
    n_points_x, n_points_y = image.shape
    u = init_u
    n = (1-u)*np.random.rand(n_points_x, n_points_y, n_clusters)
    e = (1-u-n)*np.random.rand(n_points_x, n_points_y, n_clusters)
    centers = cal_centers(image,u,n,m,e,n_clusters,win_size,init_cent,sigma)
    for i in range(iterations) :
        prev_centers=centers
        u = update_u(image,u,n_clusters,m,centers,n,e,gamma,win_size,sigma )    
        n = update_n(u, e, n_clusters) 
        e = update_e(u, n, alpha)
        centers = cal_centers(image,u,n,m,e,n_clusters,win_size,prev_centers,sigma)
        if np.max(abs(np.array(prev_centers)-np.array(centers))) <= error:
            return u
    return u 

#t, cent = fit(image,4,10,2,0.0001)


# In[107]:


#tr=fit_(image, 4, 2, 0.55, 0.0001, 20, t , cent, 30000,3,150)





