function [err] = misclasserr(Y,C)

err = sum(Y(:) ~= C(:)) / size(Y,1) / 2;
