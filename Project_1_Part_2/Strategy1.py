
# coding: utf-8

# In[1]:


from Precode import *
import numpy
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S1('6522') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# # K-MEANS

# In[4]:


def k_means_strat_1(k, cents, points):
    # next set of centroids for each iteration
    next_cents = cents
    
    # assignment list of lists - keeping track of clusters
    assign_lists = [ [] for _ in range(k) ]
    
    # stop condition variable
    stop_now = False
    
    # itr_counter
    itr_counter = 0
    
    # MAIN LOOP - stops until convergence (i.e. until assign_lists has not changed)
    while (stop_now == False):
        itr_counter = itr_counter + 1
        # set stop_now to True - if ANY point changes index in algo then set it to false
        stop_now = True
        
        # track the previous assigned list before updating it
        prev_assign_lists = [[i for i in row] for row in assign_lists] 
        
        # empty assign_list for next round
        assign_lists = [ [] for _ in range(k) ]
        
        # ASSIGNMENT
        # go through each point and assign the nearest cluster
        for point in points:
            # calculate distance of this point to each centroid
            d_cents = [ numpy.linalg.norm(diff_) for diff_ in (point - next_cents) ]
        
            # get index of the minimum distance
            min_ind = numpy.argmin(d_cents)            
            
            # get previous index for point
            prev_ind = -1
            for itr in range(k):
                in_lst = prev_assign_lists[itr]
                if len(in_lst) != 0:
                    if numpy.any(numpy.all(point == in_lst, axis=1)):
                        prev_ind = itr
                        break
                    
            if prev_ind != min_ind:
                # set stop_now to False
                stop_now = False
                
            # add this point to the respective index in the assignment of lists
            assign_lists[min_ind].append(point)
        
        # NOTE: the initial centroid points will be taken care of in this too!
        
        # UPDATE
        # calculate mean of each new clusters formed - update next_cents
        next_cents = numpy.array([ numpy.mean(in_list, axis=0) for in_list in assign_lists ])
        
    # get the final centroids ...
    fin_cents = next_cents
    # ... and final assignment lists
    fin_assign_lists = assign_lists
    
    return fin_cents, fin_assign_lists


# In[5]:


final_centroids1, final_clusters1 = k_means_strat_1(k1, i_point1, data)
final_centroids2, final_clusters2 = k_means_strat_1(k2, i_point2, data)


# In[6]:


print(final_centroids1)


# In[7]:


print(final_centroids2)


# # OBJECTIVE FUNCTION

# In[8]:


def obj_func_strat_1(k, cents, points):
    # undergo K-means strat
    final_centroids, final_clusters = k_means_strat_1(k, cents, points)
    
    # double summations requiring outer and inner sums
    outer_sum = 0
    
    for ind in range(k):
        # distance between centroid and points in cluster
        inner_sum = 0
        
        # for each point in the centroids find the distance to other points
        for point in final_clusters[ind]:
            dist= numpy.linalg.norm(final_centroids[ind] - point)
            inner_sum = inner_sum + numpy.square(dist)
        
        # sum up these distances- and add to outer_sum
        outer_sum = outer_sum + inner_sum
        
    return outer_sum


# In[9]:


loss1 = obj_func_strat_1(k1, i_point1, data)
loss2 = obj_func_strat_1(k2, i_point2, data)


# In[10]:


print(loss1)


# In[11]:


print(loss2)


# # PLOT K_Strat_1-vs-LOSS - Prints values from k=2 to 10

# In[12]:


import matplotlib.pyplot as plt


# In[13]:


print("VALUES for k ranging from 2-10: INITIALIZATION 1")
k_range = range(2,11)
finial_centroids1= []
loss_vals1 = []
for k_val in k_range:
    # get random points BUT for k=3 and 5 set them to values given
    if (k_val == 3):
        cents_rand = i_point1
    elif (k_val == 5):
        cents_rand = i_point2
    else:
        cents_rand = data[numpy.random.choice(range(len(data)), k_val, replace=False)]
    
    # compute finial centroids in each k value and random data
    fin_cents_to_add, clusts = k_means_strat_1(k_val, cents_rand, data)
    finial_centroids1.append( fin_cents_to_add )
    
    # compute loss values respectively
    loss_val_to_add = obj_func_strat_1(k_val, cents_rand, data)
    loss_vals1.append( loss_val_to_add )
    
    print("k = "+str(k_val)+", Loss = "+str(loss_val_to_add))


# In[14]:


print("VALUES for k ranging from 2-10: INITIALIZATION 2")
k_range = range(2,11)
finial_centroids2= []
loss_vals2 = []
for k_val in k_range:
    # get random points
    cents_rand = data[numpy.random.choice(range(len(data)), k_val, replace=False)]
    
    # compute finial centroids in each k value and random data
    fin_cents_to_add, clusts = k_means_strat_1(k_val, cents_rand, data)
    finial_centroids2.append( fin_cents_to_add )
    
    # compute loss values respectively
    loss_val_to_add = obj_func_strat_1(k_val, cents_rand, data)
    loss_vals2.append( loss_val_to_add )
    
    print("k = "+str(k_val)+", Loss = "+str(loss_val_to_add))


# In[16]:


fig, ax = plt.subplots()
ax.plot(k_range, loss_vals1, marker='o')
ax.plot(k_range, loss_vals2, marker='*')

ax.set(xlabel='Number of Clusters K (2-10)', ylabel='Objective Function Loss', title='K-Startegy-1 vs LOSS Graph')
ax.grid()
ax.legend(["Init-1", "Init-2"])

fig.savefig("Strat1.png")
plt.show()

