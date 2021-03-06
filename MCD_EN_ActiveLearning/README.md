## Active Learning Using MC Dropouts and Ensemble Neural Networks

***Prepare the Environment***

Create your virtual python environment using pip and install the packages from ```requirements.txt```

Enable Tensorboard Visualization with the command ```tensorboard --logdir=logs```

***Folders***
The program creates the following folders under the same execution directory

1. trained_weights: After each round, the trained NN are stored in this folder.
2. logs: The program uses this folder to write intermediate logs to visualize in tensorboard.
3. results - used to hold the final results.

***How to run:***

**Examples:**
1. Dropout-based active learining on a simple lenet5 model on MNIST dataset.
Here, we are running the lenet 5 model for 40 rounds, at each round we add 100 new samples to the training set (indicated by tk). We being with a
seed training sample size of 50 (indicated by -ss). During the uncertainty estimation, we will be using 25 instances of the network (indicated
by 25)

```
python TrainDropoutClassification.py -m lenet5 -mo test -bs 100 -tk 100 -ss 50 -rt 0 -r 41 -nni 25
```

2. Similarly, if you want to run the ensemble-based active leaning do:

```
python TrainEnsemble.py -m lenet5 -mo test -bs 100 -tk 100 -ss 50 -rt 0 -r 41 -nni 25
```
The meaning of command line arguments can be seen within the code (in this case TrainDropoutClassification_v3.py). You will be able to 
monitor the progress of the training using tensorboard. Currently, the program records validation accuracy and expected calibration error.
Following is an example from the tensorboard output for the above setting:

![Training progress of lenet5 on MNIST](images/tensorboard_run1.png)

Once the model has been trained, you can change the train argument in the above command to test. This will test the model on the held-out
dataset for every single round and save the output as csv. At each round, the model produces accuracy, F1, recall, expected calibration error, and support metrics for each class label. For instance, at round 40, the following is the output from the model.

|              | precision | recall | f1-score             | support              |
|--------------|-----------|--------|----------------------|----------------------|
| 0            | 0.995     | 0.992  | 0.994   | 983.0                |
| 1            | 0.994     | 0.992  | 0.993   | 1137.0               |
| 2            | 0.994     | 0.991  | 0.992   | 1035.0               |
| 3            | 0.991     | 0.988  | 0.989   | 1013.0               |
| 4            | 0.991     | 0.991  | 0.991   | 982.0                |
| 5            | 0.985     | 0.983  | 0.984   | 894.0                |
| 6            | 0.988     | 0.993  | 0.991   | 953.0                |
| 7            | 0.994     | 0.992  | 0.993   | 1030.0               |
| 8            | 0.990     | 0.987  | 0.989   | 977.0                |
| 9            | 0.980     | 0.992  | 0.986  | 996.0                |
| accuracy     | 0.9908    | 0.990  | 0.9908               | 0.9908               |
| macro avg    | 0.990     | 0.990  | 0.990   | 10000.0              |
| weighted avg | 0.990     | 0.990  | 0.990   | 10000.0              |
| ece          | 0.006     | 0.006  | 0.006 | 0.006 |