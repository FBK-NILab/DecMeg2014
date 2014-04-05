import numpy as np
from scipy.io import loadmat



if __name__ == '__main__':

    np.random.seed(0)
    subjects = range(17, 24)

    f = open('random_submission.csv','w')
    print >> f, 'Id,Prediction'
    for subject in subjects:
        filename = 'data/test_subject'+str(subject)+'.mat'
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        ids = data['Id']
        prediction = (np.random.rand(len(ids)) > 0.5).astype(np.int)
        for i, ids_i in enumerate(ids):
            print >> f, str(ids_i) + ',' + str(prediction[i])

    f.close()
                      
