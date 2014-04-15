% DecMeg2014 example code.
% Simple prediction of the class labels of the test set by:
%   - pooling all the triaining trials of all subjects in one dataset.
%   - Extracting the MEG data in the first 500ms from when the stimulus starts.
% - Using a linear classifier (logistic regression).
clear all;
disp('DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain');
subjects_train = 1:16;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));
% We throw away all the MEG data outside the first 0.5sec from when
% the visual stimulus start:
tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
X_train = [];
y_train = [];
X_test = [];
ids_test = [];
% Crating the trainset.
disp('Creating the trainset.');
for i = 1 : length(subjects_train)
    filename = sprintf('/home/mosi/Project/bm2014/data/train_subject%02d.mat',subjects_train(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    yy = data.y;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('yy: %d trials',size(yy,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_train = [X_train;features];
    y_train = [y_train;yy];
end

% Crating the testset.
disp('Creating the testset.');
subjects_test = 17:23;
for i = 1 : length(subjects_test)
    filename = sprintf('/home/mosi/Project/bm2014/data/test_subject%02d.mat',subjects_test(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    ids = data.Id;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('Ids: %d trials',size(ids,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_test = [X_test;features];
    ids_test = [ids_test;ids];
end

% Your training code should be here:
disp('Training the classifier ...')
[BFinal,FitInfoFinal] = lasso(X_train,single(y_train),'Lambda',0.005,'Alpha',0.9);

% Testing the trained classifier on the test data
y_pred = [ones(size(X_test,1),1) X_test] * [FitInfoFinal.Intercept;BFinal];
y_pred_thresholded = zeros(size(y_pred));
y_pred_thresholded(y_pred>=median(y_pred))= 1;

% Saving the results in the submission file:
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(y_pred_thresholded)
    fprintf(f,'%d,%d\n',ids_test(i),y_pred_thresholded(i));
end
fclose(f);
disp('Done.');