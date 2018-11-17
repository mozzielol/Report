import numpy as np
from copy import deepcopy

class Info(object):

	def __init__(self,**args):
		self.args = args
		self.args['fisher'] = None
		self.args['index'] = None
		self.args['data'] = []
		self.args['label'] = []
		self.args['data_num'] = 1

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
        

	def update_fisher(self,value):
		fisher = self.args['fisher']
		sum = 0
		if fisher is None:
			self.args['fisher'] = value
			self.args['index'] = deepcopy(value)
			for v in range(len(value)):
				self.args['index'][v] = np.ones(value[v].shape)
		else:
			for i in range(len(fisher)):
				indi = np.where(fisher[i] < value[i])
				fisher[i][indi] = value[i][indi]
				self.args['index'][i][indi] = self.args['data_num']

				div = 1
				for s in self.args['index'][i].shape:
					div *= s
				sum += len(indi[0]) * 100 / div
				print('{} percent fisher information is updated'.format(str(len(indi[0]) * 100 / div)))
			print(len(self.args['index']),len(self.args['index'][0]))
			self.args['fisher'] = fisher
		self.args['data_num'] += 1

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
