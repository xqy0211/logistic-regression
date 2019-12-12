from utils import loadDataSet, normalize, grab, store
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap
import numpy as np
import random
import argparse
import copy


def softmax(X):
	"""
	Calculate X's softmax values.
	math:	h[idx_sample, i] =  exp(X[idx_sample, i]) / Σexp(X[idx_sample, :])

	Args:
		X(matrix): the output vectors of exp(θTx) shape: [Num_sample, Num_Class]

	Return:
		h(matrix): same shape as X.
	"""
	num = np.exp(X) #numerator exp(X[i])
	denom = np.sum(num, axis=-1).reshape(-1,1) #denominator Σexp(X[:])
	h = num / denom
	return h

def gradAscent(data, label, x_graph, lr=0.01, weights= None, pre_train=False, maxIter=5000):
	"""
	gradient asecent for maximum likelihood estimation

	Args:
		data(List):
		label(List): labels corresponding to each sample.
		x_graph(array): contour for ploting.
		lr(float): learning rate (default=0.01)
		weights(matrix): initial weights
		pre_train(bool): whether pretrain or not.
		maxIter(int): the maximal iteration. (default=5000)

	Return:
		weights(matrix): tuned weights indicating the model is really good for classification.
		graph_list(List): element is the current contour
		cost_list(List): element's type == float
	"""
	dataMat, _ = normalize(data)
	labelMat, numClass = np.mat(label).T, len(set(label))
	m, n = np.shape(dataMat)
	if 1 - pre_train:
		weights = np.mat(np.random.randn(n, numClass))
	mask = np.mat(np.zeros((m, numClass)))
	last_loss=0; cur_loss=0
	graph_list = []; cost_list = []
	for class_ in range(numClass):
		mask[:, class_] = labelMat == class_
	for i in range(maxIter):
		h = softmax(dataMat * weights)
		cur_loss = np.trace(mask.T *-np.log(h)) / m
		if i % 50 ==0 and i!=0:
			print('Iteration {} Current Loss {:.6f}'.format(i,cur_loss))
			y_graph = np.argmax(softmax(x_graph*weights), axis=1)
			s = int(np.sqrt(y_graph.shape[0]))
			graph_list.append(y_graph.reshape(s, s))
			cost_list.append(cur_loss)
		if i > 0.5 * maxIter and np.abs(cur_loss - last_loss) < 1e-4:
			print('Stop Early')
			return weights, graph_list, cost_list
		error = mask - h
		update = dataMat.T * error
		weights += lr * update
		last_loss = cur_loss
	
	return weights, graph_list, cost_list
	
def stocGradAscent(data, label, x_graph, lr=0.2, weights= None, pre_train=False, maxIter=100):
	"""
	stochastic gradient asecent for maximum likelihood estimation

	Args:
		data(List):
		label(List): labels corresponding to each sample.
		x_graph(array): contour for ploting.
		lr(float): learning rate (default=0.2)
		weights(matrix): initial weights
		pre_train(bool): whether pretrain or not.
		maxIter(int): the maximal iteration. (default=100)

	Return:
		weights(matrix): tuned weights indicating the model is really good for classification.
		graph_list(List): element is the current contour
		cost_list(List): element's type == float
	"""
	dataMat, _ = normalize(data)
	labelMat, numClass = np.mat(label).T, len(set(label))
	m, n = np.shape(dataMat)
	if 1 - pre_train:
		weights = np.mat(np.random.randn(n, numClass))
	mask = np.mat(np.zeros((m, numClass)))
	last_loss=0; cur_loss=0
	graph_list = []; cost_list = []
	for class_ in range(numClass):
		mask[:, class_] = labelMat == class_
	for j in range(maxIter):
		dataIndex = list(range(m))
		cur_iter = j * m
		for i in range(m):
			h = softmax(dataMat * weights)
			cur_loss = np.trace(mask.T * -np.log(h)) / m
			if (cur_iter + i) % 100 ==0 and (cur_iter + i) != 0:
				print('Iteration {} Current Loss {}'.format(cur_iter+i,cur_loss))
				y_graph = np.argmax(softmax(x_graph*weights), axis=1)
				s = int(np.sqrt(y_graph.shape[0]))
				graph_list.append(y_graph.reshape(s, s))
				cost_list.append(cur_loss)
			if np.abs(cur_loss-last_loss) < 1e-4:
				print('Stop Early')
				return weights, graph_list, cost_list
			randIndex = int(random.uniform(0, len(dataIndex)))
			index = dataIndex[randIndex]
			error = mask[index] - h[index] 
			update = dataMat[index].T * error
			weights += lr * update
			del dataIndex[randIndex]
	return weights, graph_list, cost_list

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
	h = softmax(dataMat * weights)
	y = np.argmax(h, axis=1)
	accuracy = np.sum(y==labelMat) / m
	print('Accuracy {:.2f}%'.format(accuracy * 100))
	return accuracy

def plotCurve(data, label, x_grid, g_list, cost_list, mode):
	"""
	plot the boundary and cost curve.

	Args:
		data(List):
		label(List):
		x_grid(List): contour for ploting.
		g_list(List): the recorded contours.
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
	#use plot instead of scatter due to ax.contourf is able to cover scatter dots.
	ax1.plot(np.array(xcord1) * fs[1] + fs[0], np.array(ycord1) * fs[3] + fs[2], 'g+', markersize=6,label='Admitted')
	ax1.plot(np.array(xcord2) * fs[1] + fs[0], np.array(ycord2) * fs[2] + fs[3], 'ro', markersize=6, label='Not Admitted')
	ax1.legend(loc='upper left')
	ax1.xaxis.set_major_locator(MultipleLocator(5)); ax1.yaxis.set_major_locator(MultipleLocator(5))
	custom_cmap = ListedColormap(['#fafab0','#a0faa0'])#'#9898ff'])

	ax2.set_title('Cost'); ax2.set_ylim(min(cost_list)*0.95, max(cost_list) * 1.01)
	ax2.set_xlabel('Iteration /50') if mode=='GD' else ax2.set_xlabel('Iteration /100') 
	ax2.set_ylabel('Cost')
	xcord_ax2 = []; ycord_ax2 = []
	for i in range(len(g_list)):
		plt.pause(0.2)
		ax2.set_xlim(-1, i);xcord_ax2.append(i);ycord_ax2.append(cost_list[i])
		ax2.scatter(i, cost_list[i], c='r', marker='o')
		try:
			ax2.lines.remove(line_ax2[0])
		except:
			pass
		ax1.contourf(x_grid[0] , x_grid[1], g_list[i], cmap=custom_cmap)
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

	Note that: --plot and --pretrain must have corresponding files (w_file, glist_file, clist_file) stored before.
	"""
	parser = argparse.ArgumentParser(description='softmaxRegression')
	parser.add_argument('--mode', type=str, default='SGD')
	parser.add_argument('--plot', action='store_true', default=False,
						help='whether only plot or not (default=False)')
	parser.add_argument('--pretrain', action='store_true', default=False,
						help='use pretrained weights or initial weights (default=False)')
	parser.add_argument('--save', action='store_true', default=False,
						help='save weights, weights_history and cost_history (default=False)')
	args = parser.parse_args()
	#files' path of weights, glist and clist.
	w_file = 'weights_Soft_GD.txt' if args.mode == 'GD' else 'weights_Soft_SGD.txt'
	glist_file = 'g_list_Soft_GD.txt' if args.mode == 'GD' else 'g_list_Soft_SGD.txt'
	clist_file = 'c_list_Soft_GD.txt' if args.mode == 'GD' else 'c_list_Soft_SGD.txt'
	
	dataList, labelList = loadDataSet()
	_, fs = normalize(dataList)
	#for contour
	x_grid = np.meshgrid(
		np.linspace(15, 65, 400).reshape(-1, 1),
		np.linspace(40, 90, 400).reshape(-1, 1),
		)
	x = np.c_[x_grid[0].ravel(), x_grid[1].ravel()]
	x_bias = np.c_[np.ones((len(x), 1)), x]
	for i in range(2):
		x_bias[:,i+1] = (x_bias[:,i+1] - fs[i*2]) / fs[i*2+1]
	if not args.plot:
		if args.pretrain:
			weights = grab(w_file)
			if args.mode =='GD':
				weights,g_list, cost_list = gradAscent(dataList, labelList, x_bias, weights=weights, pre_train=True)
			elif args.mode=='SGD':
				weights, g_list, cost_list = stocGradAscent(dataList, labelList, x_bias, weights=weights, pre_train=True)
		else:
			if args.mode=='GD':
				weights, g_list, cost_list = gradAscent(dataList, labelList, x_bias)
			elif args.mode=='SGD':
				weights, g_list, cost_list = stocGradAscent(dataList, labelList, x_bias)
			if args.save:
				store(weights, w_file)
				store(g_list, glist_file)
				store(cost_list, clist_file)
	else:
		try:
			weights = grab(w_file)
			g_list = grab(glist_file)
			cost_list = grab(clist_file)
		except:
			raise IOError('No pretrained file. Please python xxx.py --mode GD/SGD --save first')
	
	classify(dataList, labelList, weights)
	plotCurve(dataList, labelList, x_grid, g_list, cost_list, args.mode)
