### ---------- ###
"""
Please compile your own evaluation code based on the training code 
to evaluate the trained network.
The function name and the inputs of the function have been predifined and please finish the remaining part.
"""
def evaluate(net, images, labels):
    acc = 0    
    loss = 0
    batch_size = 1

    pass
    for batch_index in range(0, images.shape[0], batch_size):
        """
        Please compile your main code here.
        """
        # image x and label y
        x = images[batch_index]
        y = labels[batch_index]
        
        # run image through net
        for l in range(net.lay_num):
            output = net.layers[l].forward(x)
            x = output
        # compute loss for this image and increment
        loss += cross_entropy(output, y)
        if np.argmax(output) == np.argmax(y):
            acc += 1
    
    # divide loss and accuracy by number of samples
    loss = loss/images.shape[0]
    acc = acc/images.shape[0]
    return acc, loss