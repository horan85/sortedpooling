import numpy as np
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
#read data and display data

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # we will use one hot encoding, every outputclass is a separate dimension
#the parameters of the algorithm
BatchLength= 4  #batches of 32 images are processed and averaged out
Size=[28,28,1]  #size of the input image
NumIteration=1001 #we will run just a few itereations- you can run it longer at home
LearningRate=1e-4 #initial learning rate
NumClasses = 10 #number of possible output classes
EvalFreq = 100 # we will evaluate at every 1000 step
NumKernels=[8,32,64]

tf.reset_default_graph() #reset the graph
InputData = tf.placeholder(tf.float32, [BatchLength]+Size )  #input images
OneHotLabels = tf.placeholder(tf.int32, [BatchLength, NumClasses]) #the expected outputs, labels


def MaxPoolingRef(In,K,S):
	return tf.nn.max_pool(In,ksize=[1,K[0],K[1],1],strides=[1,S[0],S[1],1],padding='VALID')

def AvgPoolingRef(In,K,S):
	return tf.nn.avg_pool(In,ksize=[1,K[0],K[1],1],strides=[1,S[0],S[1],1],padding='VALID')


def MaxPoolingOwn(In,K,S):
      N=In.get_shape()[3]
      Patches=tf.extract_image_patches(In, ksizes=[1]+K+[1], strides=[1]+S+[1], rates=[1,1,1,1],padding='VALID')
      PatchShape=Patches.get_shape()
      MaxValues,MaxIndices=tf.nn.top_k(tf.transpose( tf.reshape(Patches,[BatchLength,PatchShape[1],PatchShape[2],KernelSize[0]*KernelSize[1],N]),[0,1,2,4,3]),1)
      Pooled=MaxValues[:,:,:,:,0]
      return Pooled
    
def KthPooling(In,K,S,M):
      N=In.get_shape()[3]
      Patches=tf.extract_image_patches(In, ksizes=[1]+K+[1], strides=[1]+S+[1], rates=[1,1,1,1],padding='VALID')
      PatchShape=Patches.get_shape()
      MaxValues,MaxIndices=tf.nn.top_k(tf.transpose( tf.reshape(Patches,[1,PatchShape[1],PatchShape[2],KernelSize[0]*KernelSize[1],N]),[0,1,2,4,3]),M)
      Pooled=MaxValues[:,:,:,:,M]
      return Pooled

def SortedPooling(In,K,S,M):
      N=In.get_shape()[3]
      Patches=tf.extract_image_patches(In, ksizes=[1]+K+[1], strides=[1]+S+[1], rates=[1,1,1,1],padding='VALID')
      PatchShape=Patches.get_shape()
      MaxValues,MaxIndices=tf.nn.top_k( tf.transpose( tf.reshape( Patches, [int(BatchLength),int(PatchShape[1]),int(PatchShape[2]),int(KernelSize[0]*KernelSize[1]),int(N)]),[0,1,2,4,3]),M)

      PoolWeights =tf.get_variable("PoolW", dtype=tf.float32, initializer=tf.constant(  np.ones([1,1,1,1,M,1])/float(M),dtype=tf.float32)  )
      PoolWeights =tf.tile( PoolWeights,  [int(BatchLength),int(PatchShape[1]),int(PatchShape[2]),int(N),1,1] )
      MaxValues=tf.expand_dims(MaxValues,4)
      Pooled=tf.squeeze(tf.matmul(MaxValues, tf.nn.softmax(PoolWeights,4) ),[4,5])
      return Pooled

def SortedPoolingPerChannel(In,K,S,M):
      N=In.get_shape()[3]
      Patches=tf.extract_image_patches(In, ksizes=[1]+K+[1], strides=[1]+S+[1], rates=[1,1,1,1],padding='VALID')
      PatchShape=Patches.get_shape()
      MaxValues,MaxIndices=tf.nn.top_k(tf.transpose( tf.reshape(Patches,[BatchLength,PatchShape[1],PatchShape[2],KernelSize[0]*KernelSize[1],int(N)]),[0,1,2,4,3]),M)
      PoolWeights =tf.get_variable("PoolW", dtype=tf.float32, initializer=tf.constant(  np.ones([1,1,1,int(N),int(M),1])/float(M),dtype=tf.float32)  )
      PoolWeights =tf.tile(PoolWeights,  [BatchLength,PatchShape[1],PatchShape[2],1,1,1] )
      MaxValues=tf.expand_dims(MaxValues,4)
      Pooled=tf.squeeze(tf.matmul(MaxValues, tf.nn.softmax(PoolWeights,4) ),[4,5])
      return Pooled



CurrentInput=InputData
CurrentFilters=Size[2]
LayerNum=0
# a loop which creates all layers
for N in NumKernels:
    with tf.variable_scope('conv'+str(LayerNum)):
      LayerNum+=1
      #variables that we want to optimize
      W =tf.get_variable('W', [3,3,CurrentFilters,N])
      Bias = tf.get_variable('Bias', [N],initializer=tf.constant_initializer(0.0))
      #convolution
      ConvResult = tf.nn.conv2d(CurrentInput,W,strides=[1,1,1,1], padding='VALID')
      CurrentFilters=N
      #we adda bias
      ConvResult = tf.add(ConvResult,Bias)
      print(ConvResult)
      # relu
      ReLU=tf.nn.relu(ConvResult)
      #pool
      #Pooled=tf.nn.max_pool(ReLU,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
	
      KernelSize=[3,3]
      Stride=[1,1]
      Pooled=SortedPoolingPerChannel(ReLU,KernelSize,Stride,4)
      #Patches=tf.extract_image_patches(ReLU, ksizes=[1]+KernelSize+[1], strides=[1]+Stride+[1], rates=[1,1,1,1],padding='VALID')
      #PatchShape=Patches.get_shape()
      #MaxValues,MaxIndices=tf.nn.top_k(tf.transpose( tf.reshape(Patches,[BatchLength,PatchShape[1],PatchShape[2],KernelSize[0]*KernelSize[1],N]),[0,1,2,4,3]),4)
      #Pooled=MaxValues[:,:,:,:,3]
      #print MaxValues
      #Pooled=tf.squeeze(MaxValues,-1)

      CurrentInput=Pooled
      print(Pooled)
#we have generated feature maps, we will use a fully connected layer with ten neurons, one for each class
#the response of these neruons will represent how "strongly" the element belong to this class
with tf.variable_scope('FC'):
	    CurrentShape=CurrentInput.get_shape()
	    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])
	    FC = tf.reshape(CurrentInput, [-1, FeatureLength])
	    W = tf.get_variable('W',[FeatureLength,NumClasses])
	    FC = tf.matmul(FC, W)
	    Bias = tf.get_variable('Bias',[NumClasses])
	    FC = tf.add(FC, Bias)
print(FC)

#we use softmax to normalize the outputs of the network 
#sotfmax camoes from the logistic regression and is e^i / sum(e^j) for all i and j
#this will normlaize all the values between zero and one and the sum of values will be 1

#corss entropy measures similarity between two distributions. If cross entropy is zero, the two distributions are the same
with tf.name_scope('loss'):
	    Loss = tf.reduce_mean( tf.losses.softmax_cross_entropy(OneHotLabels,FC)  )

with tf.name_scope('optimizer'):    
	    #Use ADAM optimizer this is currently the best performing training algorithm in most cases
	    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)
        #Optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):	  
	    CorrectPredictions = tf.equal(tf.argmax(FC, 1), tf.argmax(OneHotLabels, 1))
	    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))


Init = tf.global_variables_initializer()
with tf.Session() as Sess:
	Sess.run(Init)
	
	Step = 1
	# Keep training until reach max iterations - other stopping criterion could be added
	while Step < NumIteration:
		UsedInBatch= random.sample( range(mnist.train.images.shape[0]), BatchLength)
		batch_xs = mnist.train.images[UsedInBatch,:]
        	batch_ys = mnist.train.labels[UsedInBatch,:]
        	batch_xs=np.reshape(batch_xs,[BatchLength]+Size)
        	_,Acc,L = Sess.run([Optimizer, Accuracy, Loss], feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
	        if (Step%100)==0:
	        	print("Iteration: "+str(Step))
	        	print("Accuracy:" + str(Acc))
	        	print("Loss:" + str(L))
			for v in tf.global_variables():
				if v.name == "conv0/PoolW:0":
					print v.eval()
		
       		#independent test accuracy
        	if (Step%EvalFreq)==0:
			SumAcc=0.0
			S=0
			for i in range(0,mnist.test.images.shape[0],BatchLength):
				batch_xs = mnist.test.images[i:(i+BatchLength),:]
				batch_ys = mnist.test.labels[i:(i+BatchLength),:]
				batch_xs=np.reshape(batch_xs,[BatchLength]+Size)
                		batch_ys=np.reshape(batch_ys,[BatchLength,NumClasses])
				a = Sess.run(Accuracy, feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
				SumAcc+=a 
				S+=1 
			print("Independent Test set: "+str(float(SumAcc)/float(S)))
        	Step+=1	      
	
