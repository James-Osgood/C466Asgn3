function [model] = learn(X, y)
% X is a txn matrix.
% y is a tX1 vector of target labels.
% model is a learner represention.
    [t, n] = size(X);
    
    Y = class_to_vec(y);
    sigma = 280;
    K = gausskernel(X, X, sigma);
    beta = 0.5;
    
    [Lambda, obj, iter] = svm_fast(K, Y, beta);
    
    model = [X Y Lambda];

end

function [Y] = class_to_vec(y)
% class is the classisfication from the class_set.
% class_set is the set {p, c, b, d, h, s, t}.
% pred is the 8x1 standard basis vector corresponding to the classisfication.
    
    t = length(y);
    Y = zeros(t, 7);
    
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
    TOL = 1e-2; % old val = 1e-8
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

function [K] = gausskernel(X1,X2,sigma)

    distance = repmat(sum(X1.^2,2),1,size(X2,1)) ...
        + repmat(sum(X2.^2,2)',size(X1,1),1) ...
        - 2*X1*X2';

    K = exp(-distance/(2*sigma^2));

end

