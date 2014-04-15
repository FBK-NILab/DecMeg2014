% DecMeg2014 example code.
% Random prediction of the class labels of the test set by:
subject_test = 17:23;
filename_submission = 'random_submission.csv';
disp(strcat('Creating sumission file',filename_submission,'...'));
f = fopen(filename_submission,'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(subject_test)
    filename = strcat('data/test_subject',num2str(subject_test(i)),'.mat');
    disp(strcat('Loading ',filename,':'));
    data = load(filename);
    ids = data.Id;
    s = length(ids);
    disp(strcat(num2str(s),' trials.'));
    prediction = round(rand(1,s));
    for j = 1 : s
        fprintf(f,'%d,%d\n',ids(j),prediction(j));
    end
end
fclose(f);
disp('Done.');