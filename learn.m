function [model] = learn(X, y)
% X is a txn matrix.
% y is a tX1 vector of target labels.
% model is a learner represention.
    [t, n] = size(X);

%      Y = zeros(t, 7);
%      
%      for s = 1:t
%          Y(s, :) = class_to_vec(y(s));
%      end
    
    Y = class_to_vec(y);
    K = X*X';
    lambda = 1;
    beta = 1 / (2*lambda);
    
    [Lambda, obj, iter] = svm_fast(K, Y, beta);
    
    model = [X Y Lambda];
    
%      Z0 = zeros(t, 7);
%      
%      
%      
%      c = -0.1;
%      phi = @(Lambda)dual_multi_svm(Lambda, K, Y, beta, c);
%      
%      I = eye(t, 7);
%      g = @(Lambda)(Lambda*I + I);
%      
%      ub = zeros(t, 7);
%      
%      % Something is not the right size here ...
%      [Z, obj, info, iter, nf, lambda_] = sqp(Z0, phi, g, [], [], ub, [], []);
%      
%      W = Z(1:n, :);
%      b = Z(n+1, :);
%      
%      model = W;
    
%      norm(X, 'fro')^2; % Frobenius norm squared.

end

function [Y] = class_to_vec(y)
% class is the classisfication from the class_set.
% class_set is the set {p, c, b, d, h, s, t}.
% pred is the 8x1 standard basis vector corresponding to the classisfication.
    
    t = length(y);
    Y = zeros(t, 7);
%      k = 7;
%      classes = 'pcbdhst';
%      
%      
%      for j = 1:k
%          Y(find(y==class(j)), j) = 1;
%      end
    
    for i = 1:t
        class = y(i);
        
        if class == 'p'
            Y(i, :) = [1 0 0 0 0 0 0];
        elseif class == 'c'
            Y(i, :) = [0 1 0 0 0 0 0];
        elseif class == 'b'
            Y(i, :) = [0 0 1 0 0 0 0];
        elseif class == 'd'
            Y(i, :) = [0 0 0 1 0 0 0];
        elseif class == 'h'
            Y(i, :) = [0 0 0 0 1 0 0];
        elseif class == 's'
            Y(i, :) = [0 0 0 0 0 1 0];
        else % class == 't'
            Y(i, :) = [0 0 0 0 0 0 1];
    end

end

function [Lambda,obj,iter] = svm_fast(K,Y,beta)
%
% Author: Dale Schuurmans
%
% Simple algorithm for multiclass SVM
%
% Solves 
% max_{Lambda>=0,sum(Lambda,2)==ones} 
%           t - tr(Y'Lambda) - tr((Y-Lambda)'K(Y-Lambda))/(2beta)

% constants
    [t,k] = size(Y);
    maxiters = 1000*t;
    TOL = 1e-8;
    P = eye(k) - ones(k)/k;
    onesk = ones(1,k);

    % initialize
    Lambda = (K + TOL*eye(t)) \ ((K - beta*eye(t))*Y);
    ProjLambda = Lambda*P + 1/k;
    M = ones(t,k);
    Mnew = ProjLambda >= 0;
    while sum(Mnew(:) < M(:)) > 0
        M = M & Mnew;
        sumM = sum(M,2);
        Lambdam = sum(Lambda.*M,2)./sumM;
        ProjLambda = (Lambda - Lambdam(:,onesk) + 1./sumM(:,onesk)).*M;
        Mnew = ProjLambda >= 0;
    end
    Grad = -Y - K*(ProjLambda - Y)/beta;
    obj = t - trace(Y'*ProjLambda) - trace((Y-ProjLambda)'*K*(Y-ProjLambda))/2/beta;
    Lambda = ProjLambda;

    %%% main loop %%%
    for iter = 1:maxiters

        % compute all rowwise improvements

            % first have to project the gradient onto all the critical constraints
        LambdaNZ = Lambda > TOL;
        ProjGrad = Grad*P;
        M = ones(t,k);
        Mnew = LambdaNZ | ProjGrad >= 0;
        while sum(Mnew(:) < M(:)) > 0
            M = M & Mnew;
            Gradm = sum(Grad.*M,2)./sum(M,2);
            ProjGrad = (Grad - Gradm(:,onesk)).*M;
            Mnew = LambdaNZ | ProjGrad >= 0;
        end

            % then compute locally optimal steps
        PG = trace(ProjGrad'*Grad);
        if PG < TOL, break, end
        PKP = trace(ProjGrad'*K*ProjGrad);
        stepopt = beta*PG/PKP;
        U = Lambda./abs(-ProjGrad.*(ProjGrad < 0));
        step = min(stepopt,min(U(:)));
        Lambdaplus = Lambda + step*ProjGrad;
        Lambdaplus = max(0,Lambdaplus);
        improvement = step*PG - step^2*PKP/2/beta;

        % check termination
        if improvement < TOL, break, end

        % greedy update
        Lambda = Lambdaplus;
        Grad = -Y - K*(Lambda - Y)/beta;
        obj = obj + improvement;
        
end

function [Lambda,obj] = svm_quadprog(K,Y,beta)
%
% Author: Dale Schuurmans
%
% Solves 
% max_{Lambda>=0,sum(Lambda,2)==ones} 
%           t - tr(Y'Lambda) - tr((Y-Lambda)'K(Y-Lambda))/(2beta)

    [t,k] = size(Y);
    H = kron(eye(k,k),K)/beta;
    y = Y(:);
    f = y - H*y;
    Aeq = kron(ones(1,k),eye(t,t));
    beq = ones(t,1);
    lb = zeros(t*k,1);
    ub = Inf([t*k 1]);

    % modified for Octave:
    % [lambda,tmp,flag] = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    [lambda, tmp, flag] = qp([], H, f, Aeq, beq, lb, up, [], [], []);
    
    Lambda = reshape(lambda,t,k);
    obj = t - y'*H*y/2 - tmp;
    
end

function [Y] = indmax(Z)
%
% Author: Dale Schuurmans
%

    [t,k] = size(Z);
    [maxZ,I] = max(Z');
    Y = zeros(t,k);
    Y(sub2ind([t k],1:t,I)) = ones(t,1);

end

function [L] = dual_multi_svm(Lambda, K, Y, beta, c)
% negative c is a (positive?) constant. (?=> c negative)
% beta = (1/(2*lambda))
% K = X*X'

    Lambda_Y_diff = Lambda - Y;
    L = c + trace(Lambda'*Y) + beta*trace(Lambda_Y_diff'*K*Lambda_Y_diff);
    
end
