function [results] = tester()
% results is a bunch of results gathered from running various learners on the
% data

    now() % start

    load data3.mat;
    [t, n] = size(X);
    
    now() % before learn
    
    model = learn(X, y);
    
    now() % after learn
    
    yhat = classify(X, model);
    
    results = sum(yhat == y) / t;
    
    now() % finished

end
