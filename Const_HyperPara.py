'''
Constants & Hyperparameters
'''
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16

#the following included to bridge RNN and LeNet Code with diff variable names
input_size = image_size**2
num_classes = num_labels
state_size = 2000
num_batches = 2000
num_steps = 5
image_size = 28
patch_size = 5


image_size = 28


kernelSize = 5
depth1Size = 6
depth2Size = 16
num_channels = 1

padding="SAME"
convStride = 1
poolStride = 2
poolFilterSize = 2

FC1HiddenUnit = 360
FC2HiddenUnit = 784

learningRate=1e-4