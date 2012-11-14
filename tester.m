function [results, learn_time] = tester()
% results is a bunch of results gathered from running various learners on the
% data

    load data3.mat;
    [t, n] = size(X);

    time_before = time();
    
    model = learn(X, y);
    
    learn_time = time() - time_before;
    
    yhat = classify(X, model);
    
    results = sum(yhat == y) / t;

end
