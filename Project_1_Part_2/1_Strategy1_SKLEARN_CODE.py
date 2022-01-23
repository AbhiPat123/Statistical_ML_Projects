
# coding: utf-8

# In[211]:

import numpy
from sklearn.cluster import KMeans
data = numpy.load('foo.npy', allow_pickle=True)


# In[212]:


#k1,i_point1,k2,i_point2 = initial_S1('6522') # please replace 0111 with your last four digit of your ID
#k1 = 3
#i_point1 = numpy.array([ numpy.array([5.77144223,9.04075394]), numpy.array([1.96633923,7.30845038]), numpy.array([2.97097541,2.39669382]) ])
#k2 = 5
#i_point2 = numpy.array([ numpy.array([3.75004647,4.90070114]), numpy.array([2.10606162,8.23183769]), numpy.array([2.81629029,3.1999725]), numpy.array([4.30228618,7.08489147]), numpy.array([2.69511302,5.93967352]) ])

k1 = 4
i_point1 = numpy.array([[1.52668895, 4.24557918]])
k2 = 6
i_point2 = numpy.array([[2.87448907, 2.657599]])


# In[213]:


#print(k1)
#print(i_point1)
#print(k2)
#print(i_point2)


kmeans = KMeans(n_clusters=k1, init="k-means++", n_init=1).fit(data)
print(kmeans.cluster_centers_)
print(kmeans.n_iter_)

print(kmeans.inertia_)





# # K-MEANS


# In[166]:


def eq_lists(assign_ls1, assign_ls2):
    
    for inner_list1, inner_list2 in zip(assign_ls1, assign_ls2):
        
        if len(inner_list1) != len(inner_list2):
            return False
        else:
            # for each point in list1 check if it is in list2
            pnts_check1 = [ numpy.any(numpy.all(point1 == inner_list2, axis=1)) for point1 in inner_list1 ]
            
            # for each point in list2 check if it is in list1
            pnts_check2 = [ numpy.any(numpy.all(point2 == inner_list1, axis=1)) for point2 in inner_list2 ]
            
            # if both checks have all true then return true
            return numpy.all(pnts_check1) and numpy.all(pnts_check2)


# In[199]:


def k_means_strat_1(k, cents, points):
    # centroid each for each iteration
    next_cents = cents
    
    # assignment list of lists - keeping track of clusters
    assign_lists = [ [] for _ in range(k) ]
    
    # stop condition variable
    stop_now = False
    
    # MAIN LOOP - stops until convergence (i.e. until assign_lists has not changed)
    while (~stop_now):
        # track the previous assigned list before updating it
        prev_assign_lists = assign_lists
        
        # ASSIGNMENT
        # go through each point and assign the nearest cluster
        for point in points:
            # calculate distance of this point to each centroid
            d_cents = [ numpy.linalg.norm(diff_) for diff_ in (point - next_cents) ]
        
            # get index of the minimum distance
            min_ind = numpy.argmin(d_cents)
        
            # add this point to the respective index in the assignment of lists
            assign_lists[min_ind].append(point)
        
        # NOTE: the initial centroid points will be taken care of in this too!
        
        # UPDATE
        # calculate mean of each new clusters formed - update next_cents
        next_cents = numpy.array([ numpy.mean(in_list, axis=0) for in_list in assign_lists ])
        
        # STOP CONDITION CHECK
        # compare assign_lists with prev_assign_lists - convert them to numpy arrays
        stop_now = eq_lists(prev_assign_lists, assign_lists)
    
    # get the final centroids ...
    fin_cents = next_cents
    # ... and final assignment lists
    fin_assign_lists = assign_lists
    
    return fin_cents, fin_assign_lists


# In[219]:


#final_centroids1, final_clusters1 = k_means_strat_1(k1, i_point1, data)
#final_centroids2, final_clusters2 = k_means_strat_1(k2, i_point2, data)


# In[220]:


#final_centroids2


# # OBJECTIVE FUNCTION

# In[208]:


def obj_func_strat_1(k, cents, points):
    # undergo K-means strat
    final_centroids, final_clusters = k_means_strat_1(k, cents, points)
    
    outer_sum = 0
    
    for ind in range(k):
        # distance between centroid and points in cluster
        d_clust = [ numpy.linalg.norm(diff_2) for diff_2 in (cents[ind] - points[ind]) ]
        
        # sum up these distances- and add to outer_sum
        outer_sum = outer_sum + sum(d_clust)
        
    return outer_sum
    


# In[221]:


#loss1 = obj_func_strat_1(k1, i_point1, data)
#loss2 = obj_func_strat_1(k2, i_point2, data)


# In[222]:


#loss2


# TEST

# In[224]:




