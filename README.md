# DD2424 Deep Learning in Data Science Assignments

### Four assignments on deep architectures:

#### Assignment 1: A two-layer neural network applied to CIFAR10

#### Assignment 2: A three-layer neural network with a cyclic learning rate applied to CIFAR10

#### Assignment 3: A k-layer neural network with batch normalization applied to CIFAR10

#### Assignment 4: A vanilla RNN to synthesize English text characters trained on The Goblet of Fire by J.K. Rowling

##### Project structure

- /assignment1 -- contains code and plot for the first assignment
- /assignment2 -- contains code and plot for the second assignment
- /assignment3 -- contains code and plot for the third assignment
- /assignment4 -- contains code and plot for the fourth assignment
- /instructions -- contains the instruction document for each assignment
- /reports -- contains the written reports for each each assignment

##### Data:

For the first three assignments, the CIFAR10 object detection dataset was used, which can be [downloaded here](https://www.cs.toronto.edu/~kriz/cifar.html).
For the last assignment, a .txt file containing "the Goblet of Fire" by J.K. Rowling was used.

##### Running the code:

After the CIFAR10 dataset has been downloaded under `assignment[1-3]/datasets/cifar-10-batches-py/` the scripts for assignments 1-3 can be run as follows (from their respective directories):

- `$ python3 assignment1.py`
- `$ python3 assignment1bonus.py`
- `$ python3 assignment2.py`
- `$ python3 assignment3.py`

To run the code for the fourth assignment, the text file for training (e.g. `example.txt`) has to be placed in `/assignment4/`. Afterwards, the code can be run as follows:

- `$ python3 assignment4.py example.txt`
