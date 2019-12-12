
from utils import loadDataSet, normalize, grab, store
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import random
import argparse
import copy

def sigmoid(X):
	"""
	Calculate X's sigmoid values.
	math:  h[X] = 1.0/(1+exp(-X))

	Args:
		X(matrix): 

	Return:
		h(matrix): same shape as X.
	"""
	return 1.0/ (1+np.exp(-X))

def gradAscent(data, label, lr=0.01, weights= None, pre_train=False, maxIter=5000):
	"""
	gradient asecent for maximum likelihood estimation
	Args:
		data(List):
		label(List): labels corresponding to each sample.
		lr(float): learning rate (default=0.01)
		weights(matrix): initial weights if pre_train==True
		pre_train(bool): whether pretrain or not.
		maxIter(int): the maximal iteration. (default=5000)

	Return:
		weights(matrix): tuned weights indicating the model is really good for classification.
		w_list(List): element's type == np.matrix
		cost_list(List): element's type == float
	"""
	dataMat, _ = normalize(data); labelMat= np.mat(label).T
	m, n = np.shape(dataMat)
	if 1 - pre_train:
		weights = np.mat(np.random.randn(n, 1))
	last_loss=0; cur_loss=0
	w_list = []; cost_list = []
	for i in range(maxIter):
		h = sigmoid(dataMat * weights)
		cur_loss = -(labelMat.T* np.log(h)+ (1-labelMat).T* np.log(1-h))[0,0] / m
		if i % 50 ==0 and i != 0:
			print('Iteration {} Current Loss {:.6f}'.format(i,cur_loss))
			w_list.append(copy.deepcopy(weights))
			cost_list.append(cur_loss)
		if i > 0.5 * maxIter and np.abs(cur_loss - last_loss) < 5e-6:
			print('Stop Early')
			return weights,w_list, cost_list
		error = labelMat - h
		update = dataMat.T * error
		weights += lr * update
		last_loss = cur_loss
		
	return weights,w_list, cost_list

def stocGradAscent(data, label, lr=0.2, weights= None, pre_train=False, maxIter=100):
	"""
	stochastic gradient asecent for maximum likelihood estimation

	Args:
		data(List):
		label(List): labels corresponding to each sample.
		lr(float): learning rate (default=0.2)
		weights(matrix): initial weights
		pre_train(bool): whether pretrain or not.
		maxIter(int): the maximal iteration. (default=100)

	Return:
		weights(matrix): tuned weights indicating the model is really good for classification.
		w_list(List): element's type == np.matrix
		cost_list(List): element's type == float
	"""
	dataMat, _ = normalize(data); labelMat= np.mat(label).T
	m, n = np.shape(dataMat)
	if 1 - pre_train:
		weights = np.mat(np.random.randn(n, 1))
	last_loss=0; cur_loss=0; flag=False
	w_list = []; cost_list = []
	for j in range(maxIter):
		if flag:
			break
		dataIndex = list(range(m))
		cur_iter = j * m
		for i in range(m):
			h = sigmoid(dataMat * weights)
			cur_loss = -(labelMat.T* np.log(h) + (1-labelMat).T* np.log(1-h))[0,0] / m
			if (cur_iter + i) % 100 ==0:
				print('Iteration {} Current Loss {}'.format(cur_iter+i,cur_loss))
				w_list.append(copy.deepcopy(weights))
				cost_list.append(cur_loss)
			if np.abs(cur_loss-last_loss) < 1e-4:
				print('Stop Early')
				return weights, w_list, cost_list
			randIndex = int(random.uniform(0, len(dataIndex)))
			index = dataIndex[randIndex]
			error = labelMat[index] - h[index]
			update = dataMat[index].T * error
			weights += lr * update
			del dataIndex[randIndex]
	return weights, w_list, cost_list

def classify(data, label, weights):
	"""
	Classify the dataset and get accuracy.

	Args:
		data(List):
		label(List):labels corresponding to each sample.
		weights(matrix): the model's best weights for classification.

	Return:
		accuracy(float):
	"""
	dataMat, _ = normalize(data); labelMat= np.mat(label).T
	m, n = np.shape(dataMat)
	h = sigmoid(dataMat * weights)
	y = np.float32(h > 0.5)
	accuracy = np.sum(y == labelMat) / m
	print('Accuracy {:.2f}%'.format(accuracy * 100))

def plotCurve(data, label, w_list, cost_list, mode):
	"""
	plot the boundary and cost curve.

	Args:
		data(List):
		label(List):
		w_list(List): the recorded weights. Element Type: np.matrix
		cost_list(List): the recorded costs.  Element Type: float

	Return:
	"""
	dataMat, fs = normalize(data); labelMat= np.mat(label).T
	m, n = np.shape(dataMat)
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(m):
		if labelMat[i,0] == 1.:
			xcord1.append(dataMat[i,1]); ycord1.append(dataMat[i,2])
		else:
			xcord2.append(dataMat[i,1]); ycord2.append(dataMat[i,2])
	plt.ion()
	ax1 = plt.subplot(1, 2, 1);ax2 = plt.subplot(1, 2, 2)
	ax1.set_title('Boundary');ax1.set_xlim(15, 65);ax1.set_ylim(40, 90)
	ax1.set_xlabel('exam1 score'); ax1.set_ylabel('exam2 score')
	ax1.plot(np.array(xcord1)*fs[1] + fs[0], np.array(ycord1)*fs[3]+fs[2], 'g+', markersize=6,label='Admitted')
	ax1.plot(np.array(xcord2)*fs[1] + fs[0], np.array(ycord2)*fs[3]+fs[2], 'ro', markersize=6, label='Not Admitted')
	ax1.legend(loc='upper left')
	ax1.xaxis.set_major_locator(MultipleLocator(5)); ax1.yaxis.set_major_locator(MultipleLocator(5))
	x_line = np.linspace(-0.5, 1.5, 400)

	ax2.set_title('Cost'); ax2.set_ylim(min(cost_list)*0.95, max(cost_list)*1.01)
	ax2.set_xlabel('Iteration /50') if mode == 'GD' else ax2.set_xlabel('Iteration / 100')
	ax2.set_ylabel('Cost')
	xcord_ax2 = []; ycord_ax2 = []
	for i in range(len(w_list)):
		plt.pause(0.2)
		ax2.set_xlim(-1, i);xcord_ax2.append(i);ycord_ax2.append(cost_list[i])
		ax2.scatter(i, cost_list[i], c='r', marker='o')
		try:
			ax1.lines.remove(line_ax1[0])
			ax2.lines.remove(line_ax2[0])
		except:
			pass
		cur_w = w_list[i]
		y_line = (- cur_w[0,0] - cur_w[1,0] * x_line) / cur_w[2,0]
		line_ax1 = ax1.plot(x_line*fs[1]+fs[0], y_line*fs[3]+fs[2], c='black')
		line_ax2 = ax2.plot(xcord_ax2, ycord_ax2)

	plt.ioff()
	plt.show()
	return

if __name__ == '__main__':
	"""
	Usage:
		python logsticRegression.py --mode GD or SGD 
									--plot(if only need to plot) 
									--pretrain(if have weights pretrained)
									--save(if you wanna save weights)

	Note that: --plot and --pretrain must have corresponding files (w_file, wlist_file, clist_file) stored before.
	"""
	parser = argparse.ArgumentParser(description='logsticRegression')
	parser.add_argument('--mode', type=str, default='SGD')
	parser.add_argument('--plot', action='store_true', default=False,
						help='whether only plot or not (default=False)')
	parser.add_argument('--pretrain', action='store_true', default=False,
						help='use pretrained weights or initial weights (default=False)')
	parser.add_argument('--save', action='store_true', default=False,
						help='save weights, weights_history and cost_history (default=False)')
	args = parser.parse_args()
	#files' path of weights, wlist and clist.
	w_file = 'weights_LR_GD.txt' if args.mode == 'GD' else 'weights_LR_SGD.txt'
	wlist_file = 'w_list_LR_GD.txt' if args.mode == 'GD' else 'w_list_LR_SGD.txt'
	clist_file = 'c_list_LR_GD.txt' if args.mode == 'GD' else 'c_list_LR_SGD.txt'
	
	dataList, labelList = loadDataSet()
	if not args.plot:
		if args.pretrain:
			weights = grab(w_file)
			if args.mode =='GD':
				weights,w_list, cost_list = gradAscent(dataList, labelList, weights=weights, pre_train=True)
			elif args.mode=='SGD':
				weights, w_list, cost_list = stocGradAscent(dataList, labelList, weights=weights, pre_train=True)
		else:
			if args.mode=='GD':
				weights, w_list, cost_list = gradAscent(dataList, labelList)
			elif args.mode=='SGD':
				weights, w_list, cost_list = stocGradAscent(dataList, labelList)
			if args.save:
				store(weights, w_file)
				store(w_list, wlist_file)
				store(cost_list, clist_file)
	else:
		try:
			weights = grab(w_file)
			w_list = grab(wlist_file)
			cost_list = grab(clist_file)
		except:
			raise IOError('No pretrained file. Please python xxx.py --mode GD/SGD --save first')
	
	classify(dataList, labelList, weights)
	plotCurve(dataList, labelList, w_list, cost_list, args.mode)

