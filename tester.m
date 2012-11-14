function [results] = tester()
% results is a bunch of results gathered from running various learners on the
% data

    disp("Start")

    now() % start

    load data3.mat;
    [t, n] = size(X);
    
    disp("before learn")
    
    now() % before learn
    
    model = learn(X, y);
    
    disp("after learn")
    
    now() % after learn
    
    yhat = classify(X, model);
    
    t
    results = sum(yhat == y) / t;
    
    disp("finished")
    
    now() % finished

end
