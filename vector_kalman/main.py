
from utils import create_permuted_mnist_task, load_cifar10, create_disjoint_mnist_task
from model import Model
import matplotlib.pyplot as plt


#igore the warning messages. Cause the kal drawback is slower than normal process, keras 
#will print some warning messages.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#Plot the accuracy of test data
#Parameters:
# - name: the name of the model. It will be used in label
# - acc: list of accuracy
# - data_num: which data is plotted(D1,D2 or D3)
def acc_plot(name,acc,data_num):
	plt.figure(1)
	sub = '31'+str(data_num)
	plt.subplot(sub)
	plt.title('test accuracy on {}th dataset'.format(data_num))
	plt.plot(acc,label=name)
	plt.ylabel('acc')
	plt.xlabel('training time')
	for i in range(len(acc)-1):
		plt.vlines((i+1),0,1,color='r',linestyles='dashed')
	plt.legend(loc='upper right')
	plt.subplots_adjust(wspace=1,hspace=1)
	plt.savefig('./images/permuted.png'.format(name))


#Load the drift data
#task = create_permuted_mnist_task(3)
#train_set,test_set,vali_set = create_disjoint_mnist_task()
TASK_NUM = 10
task = create_permuted_mnist_task(TASK_NUM)


#record the test accuracy. 
#Parameters:
# - model: the instance of model
# - acc_test_d1: record the accuracy of model on D1
# - acc_test_d2: record the accuracy of model on D2
# - acc_test_d3: record the accuracy of model on D3
def test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3):
    acc_test_d1.append(model.evaluate(task[0].test.images,task[0].test.labels))
    acc_test_d2.append(model.evaluate(task[1].test.images,task[1].test.labels))
    acc_test_d3.append(model.evaluate(task[2].test.images,task[2].test.labels))


# - Train the model.
# - The validation dataset is alwasy D1_test: model.val_data(X_test[:TEST_NUM],y_test[:TEST_NUM])
# - After the model is trained on each dataset, it will record the accuracy of model on test 
#	data of D1,D2 and D3 by using test_acc()
# - Return: this function will return the accuracy of D1_test,D2_test,D3_test after being trained
#	on each dataset.
def train(name):

    acc_test_d1 = []
    acc_test_d2 = []
    acc_test_d3 = []
    t1_x,t1_y = task[0].train.images,task[0].train.labels
    t2_x,t2_y = task[1].train.images,task[1].train.labels
    t3_x,t3_y = task[2].train.images,task[2].train.labels

    model = Model()
    model.val_data(task[0].validation.images,task[0].validation.labels)
    model.fit(t1_x,t1_y)
    test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
    
    if name == 'kal':
        for i in range(1,TASK_NUM):
            print('---'*10,i,'---'*10)
            model.set_fisher_data(task[i-1].validation.images[:128],task[i-1].validation.labels[:128])
            model.transfer(task[i].train.images,task[i].train.labels)
            test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        print('go back learning')
        model.set_fisher_data(task[-1].validation.images[:128],task[-1].validation.labels[:128])
        model.transfer(t1_x,t1_y,num=1)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.set_fisher_data(task[0].validation.images[:128],task[0].validation.labels[:128])
        model.transfer(t2_x,t2_y,num=2)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.set_fisher_data(task[1].validation.images[:128],task[1].validation.labels[:128])
        model.transfer(t3_x,t3_y,num=3)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)

    if name == 'nor':
        for i in range(1,TASK_NUM):
            print('---'*10,i,'---'*10)
            model.fit(task[i].train.images,task[i].train.labels)
            test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        print('go back learning')
        model.fit(t1_x,t1_y)
        model.fit(t2_x,t2_y)
        model.fit(t3_x,t3_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)

    model.save(name)
    model.plot('res',name)
	
    return acc_test_d1,acc_test_d2,acc_test_d3

def save_acc(name,d):
	import json
	for i in range(3):
		path = './logs/permuted/acc{}/{}.txt'.format(str(i+1),name)
		with open(path,'w') as f:
			json.dump(d[i],f)

if __name__ == '__main__':
	#Train the model.
    '''
    train_task, test_task = load_cifar10()
    model = Model('cnn')
    model.val_data(test_task[0][0], test_task[0][1])
    model.fit(train_task[0][0], train_task[0][1])
    model.set_fisher_data(train_task[0][0], train_task[0][1])
    model.transfer(train_task[1][0], train_task[1][1])
    '''
    '''
    train_set,test_set,vali_set = create_disjoint_mnist_task()
    model = Model()
    model.val_data(vali_set[0][0], vali_set[0][1])
    model.fit(train_set[0][0], train_set[0][1])
    model.set_fisher_data(train_set[0][0], train_set[0][1])
    model.transfer(train_set[1][0], train_set[1][1])
    '''
    
    print('--'*10,'kal','--'*10)
    kal_d1,kal_d2,kal_d3 = train('kal')
    save_acc('kal',[kal_d1,kal_d2,kal_d3])

    print('--' * 10, 'nor', '--' * 10)
    nor_d1, nor_d2, nor_d3 = train('nor')
    save_acc('nor', [nor_d1, nor_d2, nor_d3])


    acc_plot('nor',nor_d1,1)
    acc_plot('nor',nor_d2,2)
    acc_plot('nor',nor_d3,3)
    acc_plot('kal',kal_d1,1)
    acc_plot('kal',kal_d2,2)
    acc_plot('kal',kal_d3,3)
    