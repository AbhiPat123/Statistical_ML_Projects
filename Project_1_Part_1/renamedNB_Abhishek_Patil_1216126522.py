
# coding: utf-8

# In[2]:


import numpy
import scipy.io
import math
import geneNewData

# gives probabilities from a normal distribution of mean and var
def norm_prob(x, mean, var):
    # answer is n/d
    n = math.exp(-(float(x)-float(mean))**2/(2*var))
    d = (2*math.pi*float(var))**.5
    return n/d

def main():
    myID='6522'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    # TASK 1
    # TRAIN SET - CONVERT TO 2D
    # Feature 1: average brightness of each image for both trainsets (0 and 1 digits)
    ft1_train0 = numpy.array([ numpy.mean(item) for item in list(train0) ])
    ft1_train1 = numpy.array([ numpy.mean(item) for item in list(train1) ])
    # Feature 2: std dev of brightness of each image for both trainsets (0 and 1 digits)
    ft2_train0 = numpy.array([ numpy.std(item) for item in list(train0) ])
    ft2_train1 = numpy.array([ numpy.std(item) for item in list(train1) ])
    # create 2D converted data for the train set
    train0_cvrtd = numpy.concatenate((ft1_train0.reshape((-1,1)), ft2_train0.reshape((-1,1))), axis = 1)
    train1_cvrtd = numpy.concatenate((ft1_train1.reshape((-1,1)), ft2_train1.reshape((-1,1))), axis = 1)
    
    # TEST SET - CONVERT TO 2D
    # Feature 1: average brightness of each image for both test sets (0 and 1 digits)
    ft1_test0 = numpy.array([ numpy.mean(item) for item in list(test0) ])
    ft1_test1 = numpy.array([ numpy.mean(item) for item in list(test1) ])
    # Feature 2: std dev of brightness of each image for both test sets (0 and 1 digits)
    ft2_test0 = numpy.array([ numpy.std(item) for item in list(test0) ])
    ft2_test1 = numpy.array([ numpy.std(item) for item in list(test1) ])
    # create 2D converted data for the test set
    test0_cvrtd = numpy.concatenate((ft1_test0.reshape((-1,1)), ft2_test0.reshape((-1,1))), axis = 1)
    test1_cvrtd = numpy.concatenate((ft1_test1.reshape((-1,1)), ft2_test1.reshape((-1,1))), axis = 1)
    
    # TASK 2
    # CALCULATE FOLLOWING 8 VALUES
    # (No.1) Mean of feature1 for digit0
    mean_ft1_train0 = numpy.mean(ft1_train0)
    print("(No. 1): "+str(float(mean_ft1_train0)))

    # (No.2) Variance of feature1 for digit0
    var_ft1_train0 = numpy.var(ft1_train0)
    print("(No. 2): "+str(float(var_ft1_train0)))

    # (No.3) Mean of feature2 for digit0
    mean_ft2_train0 = numpy.mean(ft2_train0)
    print("(No. 3): "+str(float(mean_ft2_train0)))

    # (No.4) Variance of feature2 for digit0
    var_ft2_train0 = numpy.var(ft2_train0)
    print("(No. 4): "+str(float(var_ft2_train0)))

    # (No.5) Mean of feature1 for digit1
    mean_ft1_train1 = numpy.mean(ft1_train1)
    print("(No. 5): "+str(float(mean_ft1_train1)))

    # (No.6) Variance of feature1 for digit1
    var_ft1_train1 = numpy.var(ft1_train1)
    print("(No. 6): "+str(float(var_ft1_train1)))

    # (No.7) Mean of feature2 for digit1
    mean_ft2_train1 = numpy.mean(ft2_train1)
    print("(No. 7): "+str(float(mean_ft2_train1)))

    # (No.8) Variance of feature2 for digit1
    var_ft2_train1 = numpy.var(ft2_train1)
    print("(No. 8): "+str(float(var_ft2_train1)))
    
    # TASK 3
    # NB CLASSIFICATION TASK
    # FORMULA CALCULATIONS
    # total number of training samples
    tot = len(train0) + len(train1)

    # Calculate Prior probabilities P(y=0) and P(y=1)
    P_y_0 = 0.5
    P_y_1 = 1 - P_y_0

    # Calculate the Posterior probabilities using Normal PDf, mean, and Variance values (these are p(x_i|y) values)

    # Test 0 digit samples using features in test0_cvrtd (obviously the machine does not know these are digit 0/1)
    # For probability values given y=0
    pdf_ft1_test0_y_0 = numpy.array([ norm_prob(item[0], mean_ft1_train0, var_ft1_train0) for item in test0_cvrtd ])
    pdf_ft2_test0_y_0 = numpy.array([ norm_prob(item[1], mean_ft2_train0, var_ft2_train0) for item in test0_cvrtd ])
    # For probability values given y=1
    pdf_ft1_test0_y_1 = numpy.array([ norm_prob(item[0], mean_ft1_train1, var_ft1_train1) for item in test0_cvrtd ])
    pdf_ft2_test0_y_1 = numpy.array([ norm_prob(item[1], mean_ft2_train1, var_ft2_train1) for item in test0_cvrtd ])

    # Test 1 digit samples using features in test1_cvrtd
    # For probability values given y=0
    pdf_ft1_test1_y_0 = numpy.array([ norm_prob(item[0], mean_ft1_train0, var_ft1_train0) for item in test1_cvrtd ])
    pdf_ft2_test1_y_0 = numpy.array([ norm_prob(item[1], mean_ft2_train0, var_ft2_train0) for item in test1_cvrtd ])
    # For probability values given y=1
    pdf_ft1_test1_y_1 = numpy.array([ norm_prob(item[0], mean_ft1_train1, var_ft1_train1) for item in test1_cvrtd ])
    pdf_ft2_test1_y_1 = numpy.array([ norm_prob(item[1], mean_ft2_train1, var_ft2_train1) for item in test1_cvrtd ])

    # NOTE: in the above process the sample being Test 0 or Test 1 doesn't matter
    # All of them undergo the same process to get a prediction

    # multiply Prior probability with Posterior probabilities for both arguments
    # Given y=0
    val_test0_y_0 = P_y_0 * pdf_ft1_test0_y_0 * pdf_ft2_test0_y_0
    val_test1_y_0 = P_y_0 * pdf_ft1_test1_y_0 * pdf_ft2_test1_y_0
    # Given y=1
    val_test0_y_1 = P_y_1 * pdf_ft1_test0_y_1 * pdf_ft2_test0_y_1
    val_test1_y_1 = P_y_1 * pdf_ft1_test1_y_1 * pdf_ft2_test1_y_1

    # PREDICTIONS
    # Predict values on test0 dataset
    # first concatenate the y=0 and y=1 along axis=1
    argmax_argument_test0 = numpy.concatenate((val_test0_y_0.reshape((-1,1)), val_test0_y_1.reshape((-1,1))), axis = 1)
    # then find the index of max value along same axis i.e. axis=1
    pred_test0 = numpy.argmax(argmax_argument_test0, axis=1)
    # Predict values on test1 dataset
    # again concatenate the y=0 and y=1 along axis=1
    argmax_argument_test1 = numpy.concatenate((val_test1_y_0.reshape((-1,1)), val_test1_y_1.reshape((-1,1))), axis = 1)
    # then find the index of max value along same axis i.e. axis=1
    pred_test1 = numpy.argmax(argmax_argument_test1, axis=1)
    
    # TASK 4
    # ACCURACY FOR DIGIT 0 TEST SET
    acc_test0 = ( numpy.count_nonzero(pred_test0==0) / len(pred_test0) )
    print("Accuracy testset0 = "+str(float(acc_test0)))

    # ACCURACY FOR DIGIT 1 TEST SET
    acc_test1 = ( numpy.count_nonzero(pred_test1)/ len(pred_test1) )
    print("Accuracy testset1 = "+str(float(acc_test1)))
    
    pass
    
if __name__ == '__main__':
    main()

