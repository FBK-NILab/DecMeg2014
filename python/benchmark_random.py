"""DecMeg2014 example code.

Random prediction of the class labels of the test set by:

Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.
"""

import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':

    np.random.seed(0)
    subjects_test = range(17, 24)

    filename_submission = 'random_submission.csv'
    print "Creating submission file", filename_submission, "..."
    f = open(filename_submission,'w')
    print >> f, 'Id,Prediction'
    for subject in subjects_test:
        filename = 'data/test_subject'+str(subject)+'.mat'
        print "Loading", filename, ":",
        data = loadmat(filename, squeeze_me=True)
        ids = data['Id']
        size = len(ids)
        print size, 'trials.'
        prediction = (np.random.rand(size) > 0.5).astype(np.int)
        for i, ids_i in enumerate(ids):
            print >> f, str(ids_i) + ',' + str(prediction[i])

    f.close()
    print "Done."
