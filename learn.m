function [model] = learn(X, y)
% X is a txn matrix.
% y is a tX1 vector of target labels.
% model is the learner represention containing X, Y, and the Lambda given by
% the svm_fast() solver.

    [t, n] = size(X);
    sigma = 9.5;
    beta = 1;
%      [beta, sigma] = complex_control(X, Y);
    
    Y = class_to_vec(y);
    K = gausskernel(X, X, sigma);

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

end

function [beta, sigma] = complex_control(X, Y)
% CV-10 validation complexity control to get the best beta and sigma values.

    [t, n] = size(X);
%      sigmas = [9.5, 10.5, 11 , 11.5,  12];
    sigma = 9.5;
    betas = [0.95, 0.975, 1, 1.5, 5];
%      beta = 0.5;
    cv = 10;
    incr = floor(t/cv);
    corrects = zeros(5, 10);
    
    for i = 1:5
%          testSigma = sigmas(i);
        testBeta = betas(i);
        lb = 1;
        for j = 1:cv
            trainIndices = get_train_indices(j);
            Xtrain = X(trainIndices, :);
            Ytrain = Y(trainIndices, :);
            Ktrain = gausskernel(Xtrain, Xtrain, sigma);
            
            [Lambda, obj, iter] = svm_fast(Ktrain, Ytrain, testBeta);
            
            testIndices = (1:t);
            testIndices(trainIndices) = [];
            Xtest = X(testIndices, :);
            Ytest = Y(testIndices, :);
            Ktest = gausskernel(Xtest, Xtrain, sigma);
            
            yhat = indmax((1/testBeta) * Ktest * (Ytrain - Lambda));
            corrects(i,j) = sum(sum((yhat == indmax(Ytest))') == 7);

            lb = lb + incr;
        end
    end

    best = 0;
    bestIndex = 1;
    bestBeta = 0;
    bestSigma = 0;
    
    for i = 1:5
        sumCorrect = sum(corrects(i, :)) / (9*t);
    
%          disp('Beta: ')
%          disp(betas(i))
%          disp('Correct: ')
%          disp(sumCorrect)
    
        if sumCorrect > best
            best = sumCorrect;
            bestIndex = i;
        end
    end
    
%      bestSigma = sigmas(bestIndex);
%      disp('best sigma: ')
%      disp(bestSigma)
%      sigma = bestSigma;
    
    bestBeta = betas(bestIndex);
    disp('best beta: ')
    disp(bestBeta)
    beta = bestBeta;
    
end

function [trainIndices] = get_train_indices(j)
% j is an index between 1 and 10.
% trainIndices is an array of 210 equally spread indices that the learner
% should train on for cross validation (CV-10).

    trainIndices = ones(1, 210);
    
    for i = 0:6
        trainIndices(1, 1+i*30:(i+1)*30) = 1+i*300+30*(j-1):i*300+30*j;
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

end

function [Y] = indmax(Z)
%
% Author: Dr. Dale Schuurmans
%

    [t,k] = size(Z);
    [maxZ,I] = max(Z');
    Y = zeros(t,k);
    Y(sub2ind([t k],1:t,I)) = ones(t,1);

end

function [K] = gausskernel(X1,X2,sigma)
%
% Author: Dr. Dale Schuurmans
%

    distance = repmat(sum(X1.^2,2),1,size(X2,1)) ...
        + repmat(sum(X2.^2,2)',size(X1,1),1) ...
        - 2*X1*X2';

    K = exp(-distance/(2*sigma^2));

end
