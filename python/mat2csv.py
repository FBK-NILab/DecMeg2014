"""DecMeg2014 example code.

Convert each MAT file, officially distributed by the competition, into
a comma-separated value (CSV) file. Each line of the CSV file is a
trial. For train data, the first value is the class-label of the trial
(y), i.e. the category of the stimulus presented to the subject
(1=Face, 0=Scramble). For test data, the first value is the Id of the
trial. In both cases, the remaining values in the line are the MEG
values (X) of the multiple timeseries of that trial stored
sequentially. Each timeseries consists of 375 timepoints, i.e. 1.5sec
recorded at 250Hz. The groups of 375 values, one for each of the 306
channel/sensor, are stored sequentially, with the channels/sensors in
the same order as in the MAT file. For this reason, each line consists
of (1 + 375 x 306) = 114751 values.

The first line of the CSV file describes the 114751 fields of the
file: 'y' is the stimulus value, while 'XCCCttt' is the MEG value (X)
measured by channel/sensor CCC at time ttt, e.g. 'X010130' is what
channel/sensor 10 measured at timestep 130.

BEWARE: each MAT file becomes a ~2Gb CSV file.

Note: change the values of subjects, mat_dir and csv_dir according to
your necessity.
"""

import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':

    subjects = range(1, 24)
    mat_dir = 'data/'
    csv_dir = 'csv/'
    x_data_format = "%1.20e" # 21 digits in IEEE exponential format

    for subject in subjects:
        print "Subject", subject
        if subject < 17:
            filename_mat = mat_dir + 'train_subject%02d.mat' % subject
            filename_csv = csv_dir + 'train_subject%02d.csv' % subject
        else:
            filename_mat = mat_dir + 'test_subject%02d.mat' % subject
            filename_csv = csv_dir + 'test_subject%02d.csv' % subject

        print "Loading", filename_mat
        data = loadmat(filename_mat, squeeze_me=True)
        X = data['X']
        if subject < 17:
            y = data['y']
        else:
            y = data['Id']

        trials, channels, timepoints = X.shape
        print "trials, channels, timepoints:", trials, channels, timepoints

        print "Creating", filename_csv
        f = open(filename_csv, 'w')
        if subject < 17:
            print >> f, "y ,",
        else:
            print >> f, "Id ,",

        print "Writing CSV header."
        for j in range(channels):
            for k in range(timepoints):
                print >> f, "X%03d%03d" % (j, k),
                if (j < channels-1) or (k < timepoints-1):
                    print >>f, ",",
                else:
                    print >> f

        print "Writing trial information."
        for i in range(trials):
            if (i % 10) == 0:
                print "trial", i

            print >> f, "%d," %  y[i],
            for j in range(channels):
                for k in range(timepoints):
                    print >> f, x_data_format % X[i,j,k],
                    if (j < channels-1) or (k < timepoints-1):
                        print >> f, ",",
                    else:
                        print >> f

        f.close()
        print "Done."
        print
