import csv
import matplotlib.pyplot as plt
import numpy as np


model = 'VanillaCNN'
testFilename = ['Results/' + model + '_layer20_test.csv',
                'Results/' + model + '_layer56_test.csv',
                'Results/' + model + '_layer110_test.csv']
trainFilename = ['Results/' + model + '_layer20_train.csv',
                'Results/' + model + '_layer56_train.csv',
                'Results/' + model + '_layer110_train.csv']

def read_table(file):
    dta = []
    for line in csv.reader(file):
        dta.append(line)
    return np.asarray(dta)

def get_train_loss(dta):
    loss = []
    for l in dta[:, 1]:
        loss.append(float(l))
    return loss

def get_test_err(dta):
    err = []
    for e in dta[:, 2]:
        err.append((100.0 - float(e)) / 100.0)
    return err

test20 = get_test_err(read_table(open(testFilename[0], 'r')))
test56 = get_test_err(read_table(open(testFilename[1], 'r')))
test110 = get_test_err(read_table(open(testFilename[2], 'r')))

train20 = get_train_loss(read_table(open(trainFilename[0], 'r')))
train56 = get_train_loss(read_table(open(trainFilename[1], 'r')))
train110 = get_train_loss(read_table(open(trainFilename[2], 'r')))

# training loss resnet 20, 56, 110
plt.subplots()
plt.plot(range(164), train20, label= model + '20')
plt.plot(range(164), train56, label= model + '56')
plt.plot(range(164), train110, label=model + '110')
plt.legend()
plt.ylim([0., 2.5])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Training Loss')
plt.savefig(model + "_traing_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# testing error vanilla cnn 20, 56, 110
plt.subplots()
plt.plot(range(164), test20, label= model + '20')
plt.plot(range(164), test56, label= model + '56')
plt.plot(range(164), test110, label= model + '110')
plt.legend()
plt.ylim([0, 1])
plt.xlabel("Epoch 1 - 164")
plt.ylabel('Testing Error')
plt.savefig(model + "_testing_error.png", dpi=300, bbox_inches='tight')
plt.close()
