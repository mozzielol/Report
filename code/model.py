from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,MaxPooling2D,Conv2D
from keras.callbacks import Callback,TensorBoard
from keras import backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from information import Info


class Model(object):
	'''
	 - Initialize the model
  	 - important parameters:
  	 	- history: it will record all the accuracy on training dataset and validation dataset
	'''
	def __init__(self,type='nn'):
		self.num_classes = 10
		self.history = None
		self.epoch = 10
		self.verbose = True
		self.info = Info()

		if type == 'nn':
			self.model = Sequential()
			self.model.add(Dense(128,input_shape=(784,),activation='relu'))
			self.model.add(Dense(128,activation='relu'))
			self.model.add(Dense(self.num_classes,activation='softmax'))

		elif type == 'cnn':
			self.model = Sequential()
			self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3),activation='relu'))
			#model.add(Activation('relu'))
			self.model.add(Conv2D(32,(3, 3),activation='relu'))
			#model.add(Activation('relu'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			#model.add(Dropout(0.25))

			self.model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
			#model.add(Activation('relu'))
			self.model.add(Conv2D(64, (3,3),activation='relu'))
			#model.add(Activation('relu'))
			self.model.add(MaxPooling2D(pool_size=(2, 2)))
			#model.add(Dropout(0.25))

			self.model.add(Flatten())
			self.model.add(Dense(512,activation='relu'))
			#model.add(Activation('relu'))
			#model.add(Dropout(0.5))
			self.model.add(Dense(self.num_classes,activation='softmax'))
			#model.add(Activation('softmax'))

	


	#What data is used for validation
	def val_data(self,X_test,y_test):
		self.X_test = X_test
		self.y_test = y_test

	#train the model by normal gradient descent algorithm
	def fit(self,X_train,y_train):
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		if self.history is None:
			self.history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=self.verbose,validation_data=(self.X_test,self.y_test))
		else:
			history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=self.verbose,validation_data=(self.X_test,self.y_test))
			self.history.history['acc'].extend(history.history['acc'])
			self.history.history['val_acc'].extend(history.history['val_acc'])

	def set_fisher_data(self,X,y):
		self.info.set_fisher_data(X,y)

	'''
	 - This function is used for 'kal' algorithm.
	 - The model will calculate the gradient on D1[batch0], and never access to D1 again
	'''
	def transfer(self,X_train,y_train,num=None):
		self.info.set_train_data(X_train,y_train)
		kalman_filter = Kalman_filter_modifier(info=self.info,num=num)
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=self.verbose,callbacks=[kalman_filter],validation_data=(self.X_test,self.y_test))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])


	def save(self,name):
		import json
		with open('./logs/{}.txt'.format(name),'w') as f:
			json.dump(self.history.history,f)
		self.model.save('./models/{}.h5'.format(name))

	def evaluate(self,X_test,y_test):
		
		score=self.model.evaluate(X_test,y_test,batch_size=128)
		print("Convolutional neural network test loss:",score[0])
		print('Convolutional neural network test accuracy:',score[1])

		return score[1]

	def get_history(self):
		return self.history


	'''
	Plot the history of accuracy
	'''
	def plot(self,name,model,shift=2):
		plt.subplot(211)
		plt.title('accuracy on current training data')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		
		plt.plot(self.history.history['acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		plt.legend(loc='upper right')
		plt.subplot(212)
		plt.title('validation accuracy on original data')
		plt.plot(self.history.history['val_acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		plt.legend(loc='upper right')
		plt.subplots_adjust(wspace=1,hspace=1)
		plt.savefig('./images/{}.png'.format(name))
		display.display(plt.gcf())
		display.clear_output(wait=True)

	def enable_tensorboard(self):
		self.tbCallBack = TensorBoard(log_dir='./logs/mnist_drift/kal/',  
		histogram_freq=0,  
		write_graph=True,  
		write_grads=True, 
		write_images=True,
		embeddings_freq=0, 
		embeddings_layer_names=None, 
		embeddings_metadata=None)



class Kalman_filter_modifier(Callback):
	"""docstring for op_batch_callback"""
	def __init__(self,info=None,num=None):
		super(Kalman_filter_modifier, self).__init__()
		self.info = info
		self.num = num	
		self.X_train = self.info.args['X_train']
		self.y_train = self.info.args['y_train']

	#Calculate the Kalman Gain based on current gradients and previous gradients
	def Kal_gain(self,cur_grad,pre_grad):
		res = []
		for i in range(len(pre_grad)):
			temp = np.absolute(pre_grad[i]) / ( np.absolute(cur_grad[i])  * self.FISHER[i] + np.absolute(pre_grad[i]) )
			temp[np.isnan(temp)] = 1
			res.append(temp)
		return res

	#Calculate the gradients of model on D1[ batch_0 ]
	#It will be used in 'kal'. 
	#	|- For other algorithm, the self.pre_g will be updated. 
	#	|- For 'kal', self.pre_g will not be updated
	def on_train_begin(self,logs={}):
		G = self.info.get_value('G')
		X = self.info.args['X_fisher']
		y = self.info.args['y_fisher']
		self.info.update_fisher(self.fisher(get_weight_grad(self.model,X,y)))
		self.FISHER = self.info.get_fisher(num=self.num)
		index = self.info.get_index()
		if G is None:
			print('G is None')
			self.pre_g = get_weight_grad(self.model,X,y)
		else:
			print('G is not None!!~')
			self.pre_g = G
			g = get_weight_grad(self.model,X,y)
			for i in range(len(g)):
				self.pre_g[i][index[i]] = g[i][index[i]]
			


		
		#self.pre_w = get_weights(self.model)


	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch
		
	#At the begining of each batch, get the weights
	#if use previous knowledge, update previous gradients(self.pre_g)
	def on_batch_begin(self,batch,logs={}):
		self.pre_w = get_weights(self.model)


		

	#At the end of the batch:
	# |- Get the current weights
	# |- Calculate the gradients of model on X_train,y_train
	# |- Calculate the Kalman Gain
	# |- Kalman Filter to calculate the new weights and set_weights
	# |- Update error(self.pre_g): This will be used in 'kal' only. Other algorithm will update at the batch begin

	def fisher(self,g):
		fisher = []
		for i in g:
			temp = np.square(i)
			temp = (temp - np.min(temp))/ (np.max(temp) - np.min(temp))
			fisher.append(temp)
		return fisher
#

	def on_batch_end(self,batch,logs={}):
		self.cur_w = get_weights(self.model)
		self.cur_g = get_weight_grad(self.model,self.X_train[batch*128:(batch+1)*128],self.y_train[batch*128:(batch+1)*128])
		
		Kalman_gain = self.Kal_gain(self.cur_g,self.pre_g)
		new_w = []
		for P,Z,E,F in zip(self.pre_w,self.cur_w,Kalman_gain,self.FISHER):
			new_w.append(P + (Z-P) * E  )	
		self.model.set_weights(new_w)
		new_g = []
		for kal,g in zip(Kalman_gain,self.pre_g):
			new_g.append((1- kal) * g )

		self.pre_g = new_g

	def on_train_end(self,logs={}):
		self.info.set_value('G',self.pre_g)






