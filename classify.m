function [yhat] = classify(Xtest, model)
% Xtest is a texn matrix.
% model is produced by learn().
% yhat is a tex1 vector of classifications on the test patterns. The
% classifications are from the set {p, c, b, d, h, s, t}.
    
    class_set = ['pcbdhst']';
    [te, n] = size(Xtest);
    k = 7;
    sigma = 280;

    X = model(:, 1:n);
    Y = model(:, n+1:n+k);
    Lambda = model(:, n+k+1:n+2*k);
    
    K = gausskernel(Xtest, X, sigma);
    
    % yhat = indmax(x'*W)' =?=  max(Xtest'*W)(2)' % For non-kernel
    % yhat = indmax((1/beta)*K*(Y - Lambda)) % For kernel
    
    yhat = char(indmax(K' * (Y - Lambda)) * class_set);

end

function [K] = gausskernel(X1,X2,sigma)

    distance = repmat(sum(X1.^2,2),1,size(X2,1)) ...
        + repmat(sum(X2.^2,2)',size(X1,1),1) ...
        - 2*X1*X2';

    K = exp(-distance/(2*sigma^2));

end
