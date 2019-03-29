from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import random
from Network import Network as Net

    
def plot_image(image):
    
    plot = [image[i:i+28] for i in range(0,784,28)]
    
    plt.imshow(plot, cmap='gray', interpolation='nearest')
    plt.show()

def get_max(outs):
    
    return np.argmax(outs)

def normalise(images):
    
    factor = 1.0 / 255
    for i in range(len(images)):
        for j in range(784):
            images[i][j] = images[i][j] * factor
        
    
    print("Done Normalising images")
    return images

def test(brain, images, labels):
    
    while True:
        a = input("waiting")
        if a == "x":
            break
        i = random.randint(0, len(images) - 1)
        image = images[i]
        label = labels[i]
        brain.guess(image)
        guess = get_max(brain.output)
        plot_image(image)
        print("number =", label)
        print("guess =", guess)

brain = Net(784, [300,100], 10, ["sigmoid", "sigmoid","softmax"])
brain.initialise_weights()

mndata = MNIST('digits')
images, labels = mndata.load_training()

images = normalise(images)

N = 1000000
i = 0
chunks = 100
chunk = 0
block_size = N // chunks

target = [0 for i in range(10)]

epochs = 6
print("running")


possible_targets = []

for i in range(10):
    arr = []
    for j in range(10):
        arr.append(int(i == j))
        
    possible_targets.append(arr)


m = 1



for a in range(epochs):
    i = 0
    while i < 60000:
        image = images[i:i + m]
        label = labels[i:i + m]
        targets = [possible_targets[label[j]] for j in range(m)]
        brain.train(image, targets)
        i = i + m
        
    print(a+1, "out of 6")

mndata = MNIST('test_digits')
images, labels = mndata.load_testing()

images = normalise(images)

shit_brain =  Net(784, [16,16], 10, ["sigmoid", "sigmoid","sigmoid"])
shit_brain.initialise_weights()
count = 0
max_acc = len(images)

for i in range(max_acc):
    
    brain.guess(images[i])
    guess = get_max(brain.output)
    if guess == labels[i]:
        count += 1
    
print (100 * count / max_acc, "%")

ask = input("demonstrate?(y or n) ")
if ask == "y":
    test(brain, images, labels)
    test(shit_brain, images, labels)


    

print("Program End")
