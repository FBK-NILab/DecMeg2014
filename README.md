DecMeg2014 - Decoding The Human Brain
=====================================

This repository hosts benchmark code and other files for the DecMeg2014 competition. The competition is about decoding the human brain from magnetoencephalographic data. For further details see: https://www.kaggle.com/c/decoding-the-human-brain .

The code is available both in Python and Matlab, with minor differences.

Initially, two basic benchmarks are available:

* benchmark_random : the code loads the files of the test set, collects the IDs of the trials in the test set and creates a valid submission file with them and by creating random class labels.

* benchmark_pooling : here the underlying idea is to ignore the differences between the pattern of brain activity of the different subjects: that they are pooled together. the code loads the files of the train subjects in the train and of the test subjects in the test set. Then it creates a simple feature space by keeping only the data of the first 0.5sec from when the stimulus starts and then concatenating all the 306 timeseries into one feature vector. After a simple z-scoring of each feature, a linear classifier is trained on the train set and the class labels of the test set are predicted. A valid submission file is created from the predicted class labels.

To run the benchmarks,

1. Download and unzip the train data and the test data from the competition website. Note that not all the training data are necessary to run the benchmarks.

2. Download the code of this repository.

3. For Python: enter the "python" directory and run `python benchmark_random.py` or `python benchmark_pooling.py`. The pooling benchmark may take several minutes to run, depending on the number of input subjects you specify. The data are expected to be in "python/data/".

Each benchmark creates a file "submission.csv".

