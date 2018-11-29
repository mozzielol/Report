import numpy as np
from copy import deepcopy

class Info(object):

	def __init__(self,**args):
		self.args = args
		self.args['fisher'] = None
		self.args['index'] = None
		self.args['data'] = []
		self.args['label'] = []
		self.args['data_num'] = 0
		self.args['fisher_update_frac'] = []
		self.args['fisher_mean'] = []
		self.args['fisher_var'] = []
		self.args['fisher_frac_task'] = {}

	def get_value(self,key):
		v = None
		try:
			v = self.args[key]
		except KeyError:
			pass
		return v

	def set_value(self,key,value):
		self.args[key] = value

	def add(self,key,value):
		self.args[key].append(value)

	def delete(self,key):
		del self.args[key]

	def add_data(self,data,label):
		self.args['data'].append(data)
		self.args['label'].append(label)

	def add_fisher(self,value):
		self.args['fisher'].append(value)
    
	def set_train_data(self,X,y):
		self.args['X_train'] = X
		self.args['y_train'] = y


	def set_fisher_data(self,X,y):
		self.args['X_fisher'] = X
		self.args['y_fisher'] = y


	def update_fisher(self,value):
		self.args['data_num'] += 1
		fisher = self.args['fisher']
		frac = 100
		if fisher is None:
			self.args['fisher'] = value
			self.args['index'] = deepcopy(value)
			for v in range(len(value)):
				self.args['index'][v] = np.ones(value[v].shape)
		else:
			frac = 0
			for i in range(len(fisher)):
				indi = np.where(fisher[i] < value[i])
				fisher[i][indi] = value[i][indi]
				self.args['index'][i][indi] = self.args['data_num']

				div = 1
				for s in self.args['index'][i].shape:
					div *= s
				frac += len(indi[0]) * 100 / div
				print('{} percent fisher information is updated'.format(str(len(indi[0]) * 100 / div)))
			frac /= len(fisher)
			self.args['fisher'] = fisher

		fisher = self.args['fisher']
		mean = 0
		var = 0
		for i in range(len(fisher)):
			mean += np.mean(fisher[i])
			var += np.var(fisher[i])
		self.args['fisher_mean'].append(mean / len(fisher))
		self.args['fisher_var'].append(var / len(fisher))
		self.args['fisher_update_frac'].append(frac)
		for i in range(1,self.args['data_num']+1):
			indi = 0
			sum_neuron = 0
			for j in range(len(fisher)):
				sum_neuron += len(fisher[j])
				indi += len(np.where(self.args['index'][j] == i)[0])

			try:
				self.args['fisher_frac_task'][i].append(indi / sum_neuron)
			except:
				self.args['fisher_frac_task'][i] = [indi / sum_neuron]
		
		

	def get_fisher(self,num=None):
		if num is None:
			print('use previous fisher information')
			return self.args['fisher']
		else:
			print('nooot using previous fisher information')
			if self.args['data_num'] < num:
				raise ValueError('the dataset has not been trained')
			else:
				fisher = deepcopy(self.args['fisher'])
				for i in range(len(self.args['fisher'])):
					indi = np.where(self.args['index'][i]==num)
					fisher[i][indi] = 0
					div = 1
					for s in self.args['index'][i].shape:
						div *= s
					print('{} percent becomes 0'.format(str(len(indi[0]) * 100 / div )))
				return fisher

	def get_index(self):
		index = []
		n = 0
		for i in self.args['index']:
			t = np.where(i==self.args['data_num'])
			index.append(t)
			n += len(t)
		print('--'*10,n,'--'*10)
		return index


	def get_fisher_frac(self):
		return self.args['fisher_update_frac']

	def get_fisher_mean_var(self):
		return self.args['fisher_mean'],self.args['fisher_var']

	def frac_of_task(self,num=1):
		return self.args['fisher_frac_task']












