import numpy
import matplotlib.pyplot

data_file = open('csv/mnist_train_100.csv', 'r')
data_list = data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

if __name__ == '__main__':
    print(all_values, '\n', image_array)
