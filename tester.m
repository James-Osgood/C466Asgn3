function [results, learn_time] = tester()
% results is the accuracy of the learner and learn_time is the time it takes to
% learn the model.

    load data3.mat;
    [t, n] = size(X);

    tic();
    
    model = learn(X, y);
    
    yhat = classify(X, model);
    
    learn_time = toc();
    
    results = sum(yhat == y) / t;

end
