function [yhat] = classify(Xtest, model)
% Xtest is a texn matrix.
% model is produced by learn().
% yhat is a tex1 vector of classifications on the test patterns. The
% classifications are from the set {p, c, b, d, h, s, t}.
    
    class_set = ['pcbdhst']';
    [te, t] = size(Xtest);
    
    y = model;
    
    % yhat = indmax(x'*W)' =?=  max(Xtest'*W)(2)' % For non-kernel
    % yhat = indmax((1/beta)*K*(Y - Lambda)) % For kernel
    
    yhat = char(y*class_set);

end
