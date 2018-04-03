import csv
import matplotlib.pyplot as plt

with open('Results/CNN_layer20_test.csv') as f:
    reader = csv.reader(f)
    layer20list = list(reader)
print(layer20list)

plt.show()