function [err,Inds] = minalignerr(Y,Yhat)

t = size(Y,1);
Costs = -Yhat'*Y/t;
[Inds,negacc] = munkres(Costs);

acc = -negacc;
err = 1 -  acc;

