function [model] = learn(X, y)
% X is a txn matrix.
% y is a tX1 vector of target labels.
% model is a learner represention.
    [t, n] = size(X);

    Y = zeros(t, 7);
    
    for s = 1:t
        Y(s, :) = class_to_vec(y(s));
    end
    
    Z0 = zeros(t, 7);
    
    K = X*X';
    lambda = 1;
    beta = 1 / (2*lambda);
    c = -0.1;
    phi = @(Lambda)dual_multi_svm(Lambda, K, Y, beta, c);
    
    I = eye(t, 7);
    g = @(Lambda)(Lambda*I + I);
    
    ub = zeros(t, 7);
    
    % Something is not the right size here ...
    [Z, obj, info, iter, nf, lambda_] = sqp(Z0, phi, g, [], [], ub, [], []);
    
    W = Z(1:n, :);
    b = Z(n+1, :);
    
    model = W;
    
%      norm(X, 'fro')^2; % Frobenius norm squared.

end

function [pred] = class_to_vec(class)
% class is the classisfication from the class_set.
% class_set is the set {p, c, b, d, h, s, t}.
% pred is the 8x1 standard basis vector corresponding to the classisfication.
    
    if class == 'p'
        pred = [1 0 0 0 0 0 0];
    elseif class == 'c'
        pred = [0 1 0 0 0 0 0];
    elseif class == 'b'
        pred = [0 0 1 0 0 0 0];
    elseif class == 'd'
        pred = [0 0 0 1 0 0 0];
    elseif class == 'h'
        pred = [0 0 0 0 1 0 0];
    elseif class == 's'
        pred = [0 0 0 0 0 1 0];
    else % class == 't'
        pred = [0 0 0 0 0 0 1];

end

function [L] = dual_multi_svm(Lambda, K, Y, beta, c)
% negative c is a (positive?) constant. (?=> c negative)
% beta = (1/(2*lambda))
% K = X*X'

    Lambda_Y_diff = Lambda - Y;
    L = c + trace(Lambda'*Y) + beta*trace(Lambda_Y_diff'*K*Lambda_Y_diff);
    
end
