function [yhat] = classify(Xtest, model)
% Xtest is a texn matrix.
% model is produced by learn().
% yhat is a tex1 vector of classifications on the test patterns. The
% classifications are from the set {p, c, b, d, h, s, t}.
    
    class_set = ['pcbdhst']';
    [te, n] = size(Xtest);
    k = 7;

    X = model(:, 1:n);
    Y = model(:, n+1:n+k);
    Lambda = model(:, n+k+1:n+2*k);
    
    K = Xtest*X';
    
    % yhat = indmax(x'*W)' =?=  max(Xtest'*W)(2)' % For non-kernel
    % yhat = indmax((1/beta)*K*(Y - Lambda)) % For kernel
    
    yhat = char(indmax(K' * (Y - Lambda))' * class_set);

end
