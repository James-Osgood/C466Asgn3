function [results] = tester()
% results is a bunch of results gathered from running various learners on the
% data

    load data3.mat;

    model = learn(X, y);
    
    yhat = classify(X, model);
    
    results = sum(yhat - y);

end
