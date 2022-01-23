
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('6522') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# # KMEANS++ - Initialization

# In[4]:


def gen_init_cents(k, first_cent, points):
    # running list of centroids
    cents_list = [first_cent]
    
    # data size
    data_size = len(points)
    
    # run loop until we find k centroids
    while (len(cents_list) < k):
    
        # list storing avg distances for each point to the centroids
        avg_dist = [None] * data_size    
        
        # go through each point and calculate nearest distances
        for ind in range(data_size):            
            point = points[ind]
            
            # skip points already in the cents_list
            if point not in numpy.array(cents_list):            
                d_cents = [ numpy.linalg.norm(point-cent) for cent in cents_list ]
            
                # get the avg distance
                avg_dist[ind] = sum(d_cents)/len(cents_list)
            else:
                avg_dist[ind] = -1
            
        # find index of maximum avg distance
        max_ind = numpy.argmax(avg_dist)
        
        # get the point at max_ind
        next_cent = points[max_ind]
        
        # add this point to cents_list
        cents_list.append(next_cent)
        
    return numpy.array(cents_list)


# In[5]:


cent1 = gen_init_cents(k1, i_point1, data)
cent2 = gen_init_cents(k2, i_point2, data)


# In[6]:


print(cent1)


# In[7]:


print(cent2)


# # K-MEANS

# In[8]:


def k_means_strat_2(k, first_cent, points):
    
    # get the centroids in K-means++
    cents = gen_init_cents(k, first_cent, points)
    
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
            d_cents = [ numpy.linalg.norm(point - cent_) for cent_ in next_cents ]
        
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


# In[9]:


final_centroids1, final_clusters1 = k_means_strat_2(k1, i_point1, data)
final_centroids2, final_clusters2 = k_means_strat_2(k2, i_point2, data)


# In[10]:


print(final_centroids1)


# In[11]:


print(final_centroids2)


# # OBJECTIVE FUNCTION

# In[12]:


def obj_func_strat_2(k, cents, points):
    # undergo K-means strat
    final_centroids, final_clusters = k_means_strat_2(k, cents, points)
    
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


# In[13]:


loss1 = obj_func_strat_2(k1, i_point1, data)
loss2 = obj_func_strat_2(k2, i_point2, data)


# In[14]:


print(loss1)


# In[15]:


print(loss2)


# # PLOT K_Strat_2-vs-LOSS - Prints values from k=2 to 10

# In[16]:


import matplotlib.pyplot as plt


# In[17]:


print("VALUES for k ranging from 2-10: INITIALIZATION 1")
k_range = range(2,11)
finial_centroids1= []
loss_vals1 = []
for k_val in k_range:
    # get the FIRST CENTROID randomly BUT for k=4 and 6 set them to values given
    if (k_val == 4):
        first_cent_rand = i_point1
    elif (k_val == 6):
        first_cent_rand = i_point2
    else:
        first_cent_rand = data[numpy.random.choice(range(len(data)), 1, replace=False)[0]]
    
    # compute finial centroids in each k value and random data
    # the algorithm includes calculating the rest of k-1 centroids using STRATEGY 2
    fin_cents_to_add, clusts = k_means_strat_2(k_val, first_cent_rand, data)
    finial_centroids1.append( fin_cents_to_add )
    
    # compute loss values respectively
    loss_val_to_add = obj_func_strat_2(k_val, first_cent_rand, data)
    loss_vals1.append( loss_val_to_add )
    
    print("k = "+str(k_val)+", Loss = "+str(loss_val_to_add))


# In[18]:


print("VALUES for k ranging from 2-10: INITIALIZATION 2")
k_range = range(2,11)
finial_centroids2= []
loss_vals2 = []
for k_val in k_range:
    # get random points
    first_cent_rand = data[numpy.random.choice(range(len(data)), 1, replace=False)[0]]
    
    # compute finial centroids in each k value and random data
    fin_cents_to_add, clusts = k_means_strat_2(k_val, first_cent_rand, data)
    finial_centroids2.append( fin_cents_to_add )
    
    # compute loss values respectively
    loss_val_to_add = obj_func_strat_2(k_val, first_cent_rand, data)
    loss_vals2.append( loss_val_to_add )
    
    print("k = "+str(k_val)+", Loss = "+str(loss_val_to_add))


# In[19]:


fig, ax = plt.subplots()
ax.plot(k_range, loss_vals1, marker='o')
ax.plot(k_range, loss_vals2, marker='*')

ax.set(xlabel='Number of Clusters K (2-10)', ylabel='Objective Function Loss', title='K-Startegy-2 vs LOSS Graph')
ax.grid()
ax.legend(["Init-1", "Init-2"])

fig.savefig("Strat2.png")
plt.show()


# # PLOT DATA - along with centroids chosen (done for k=6)

# In[20]:


# plot settings - BASE plot of all points in data variable
fig2, ax2 = plt.subplots()
fig2.set_size_inches(15, 15, forward=True)
ax2.set(xlabel='X', ylabel='Y', title='DATA POINTS')
ax2.grid()

ax2.scatter(data[:,0], data[:,1], facecolors='black',alpha=.55, s=100)

plt.plot()

# running list of centroids
clst = [i_point2]

# data size
dsize = len(data)

itr_c = 1
colors = ["red", "blue", "green", "orange", "cyan", "violet"]
# plot first point in the plot
ax2.scatter(i_point2[0], i_point2[1], 500.0, color=colors[itr_c-1], alpha=.55)
ax2.annotate(itr_c, (i_point2[0], i_point2[1]), xytext=(i_point2[0]+0.1, i_point2[1]+0.1))

# run loop until we find k centroids
while (len(clst) < k2):

    # list storing avg distances for each point to the centroids
    adist = [None] * dsize

    # go through each point and calculate nearest distances
    for itr in range(dsize):
        pt = data[itr]
        
        # skip points already in the cents_list
        if pt not in numpy.array(clst):            
            dc = [ numpy.linalg.norm(pt-c_) for c_ in clst ]

            # get the avg distance
            adist[itr] = sum(dc)/len(clst)
        else:
            adist[itr] = -1

    # find index of maximum avg distance
    mind = numpy.argmax(adist)

    # get the point at max_ind
    nc = data[mind]

    # add this point to cents_list
    clst.append(nc)
    
    itr_c = itr_c+1
    # plot on plot
    if itr_c:
        print(nc)
        ax2.scatter(nc[0], nc[1], 500.0, color=colors[itr_c-1], alpha=.55)
        ax2.annotate(itr_c, (nc[0], nc[1]), xytext=(nc[0]+0.1, nc[1]+0.1))
    

