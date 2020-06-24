import numpy as np

def build_trainset(num_miniTrain,n_x,n_y):
	for i in range(2,2+num_miniTrain):
		if (i==2):
			X_train=np.loadtxt("./data/X_train_iter"+str(2)+".txt",delimiter=',')
			Y_train=np.loadtxt("./data/Y_train_iter"+str(2)+".txt",delimiter=',')
		else:
			X_train=np.concatenate((X_train,np.loadtxt("./data/X_train_iter"+str(i)+".txt",delimiter=',')))
			Y_train=np.concatenate((Y_train,np.loadtxt("./data/Y_train_iter"+str(i)+".txt",delimiter=',')))


	X_eval=np.loadtxt("./data/X_eval_iter1.txt",delimiter=',')
	Y_eval=np.loadtxt("./data/Y_eval_iter1.txt",delimiter=',')

	X_train=np.reshape(X_train,(X_train.shape[0],n_x,n_y,1),order='F')
	Y_train=np.reshape(Y_train,(X_train.shape[0],n_x,n_y,1),order='F')
	X_eval=np.reshape(X_eval,(X_eval.shape[0],n_x,n_y,1),order='F')
	Y_eval=np.reshape(Y_eval,(X_eval.shape[0],n_x,n_y,1),order='F')
	#print(X_train.shape)
	#print(X_eval.shape)
	return X_train, Y_train, X_eval, Y_eval

def build_mini_trainset(n_x,n_y):
	X_train=np.loadtxt("./data/X_train_iter2_mini.txt",delimiter=',')
	Y_train=np.loadtxt("./data/Y_train_iter2_mini.txt",delimiter=',')


	X_eval=np.loadtxt("./data/X_eval_iter1_mini.txt",delimiter=',')
	Y_eval=np.loadtxt("./data/Y_eval_iter1_mini.txt",delimiter=',')

	X_train=np.reshape(X_train,(X_train.shape[0],n_x,n_y,1),order='F')
	Y_train=np.reshape(Y_train,(X_train.shape[0],n_x,n_y,1),order='F')
	X_eval=np.reshape(X_eval,(X_eval.shape[0],n_x,n_y,1),order='F')
	Y_eval=np.reshape(Y_eval,(X_eval.shape[0],n_x,n_y,1),order='F')

	return X_train, Y_train, X_eval, Y_eval


def build_testset(n_x,n_y):
	X_test=np.loadtxt("./data/X_test_iter1_mini.txt",delimiter=',')
	Y_test=np.loadtxt("./data/Y_test_iter1_mini.txt",delimiter=',')


	X_test=np.reshape(X_test,(X_test.shape[0],n_x,n_y,1),order='F')
	Y_test=np.reshape(Y_test,(X_test.shape[0],n_x,n_y,1),order='F')

	return X_test, Y_test

