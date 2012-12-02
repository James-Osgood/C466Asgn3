clear all
addpath helpers
load data/data4a.mat

	% recover number of classes
k = max([y;ytest]) - min([y;ytest]) + 1;

	% construct indicator matrices of labels Y,Ytest from class vectors y,ytest
[t,n]= size(X);
Y = zeros(t,k);
Y(t*y + (1:t)') = ones(t,1);
[te,n] = size(Xtest);
Ytest = zeros(te,k);
Ytest(te*ytest + (1:te)') = ones(te,1);

	% plot data if you want to see what's going on
	% train indicated by '.', test indicated by 'o', colors show classes
figure
traintestind = [ [ones(t,1) zeros(t,1)]; [zeros(te,1) ones(te,1)] ];
plt([X; Xtest],[Y; Ytest],traintestind,3);

%%%%%%%%%% prelim %%%%%%%%%%

	% will compare 3 different versions of each algorithm (ls, pca, kmeans):
	% 1. standard
	% 2. kernelized (polynomial kernel)
	% 3. kernelized (gaussian kernel) 
	% so 9 different learning algorithms in total (3 supervised, 6 unsup)

	% kernels
K2 = polykernel(X,X,3);
K3 = gausskernel(X,X,0.25);
K2test = polykernel(Xtest,X,3);
K3test = gausskernel(Xtest,X,0.25);

beta1 = 1;
beta2 = 1000;
beta3 = 1;

Reps = 500;	% number random restarts for kmeans

	% need to store training objective values
objs = zeros(1,3);	% supervised ls
objp = zeros(1,3);	% unsupervised pca
objk = zeros(Reps,3);	% unsupervised kmeans, for each restart
minobjk = Inf + zeros(1,3);	% minimum kmeans objective

	% need to store training misclassification errors
errs = zeros(1,3);	% supervised ls
errp = zeros(1,3);	% unsupervised pca
errk = zeros(1,3);	% unsupervised kmeans

	% need to store test misclassification errors
errse = zeros(1,3);	% supervised ls
errpe = zeros(1,3);	% unsupervised pca
errke = zeros(1,3);	% unsupervised kmeans

%%%%%%%%%% run experiment %%%%%%%%%%

%%%%% supervised lsq %%%%%
[U1s,W1s,objs(1)] = lsq(X,Y,beta1);
[A2s,B2s,objs(2)] = lsq_kernel(K2,Y,beta2);
[A3s,B3s,objs(3)] = lsq_kernel(K3,Y,beta3);
		% train misclass errors
C1s = classify(X,W1s);
errs(1) = misclasserr(Y,C1s);
C2s = classify_kernel(K2,A2s);
errs(2) = misclasserr(Y,C2s);
C3s = classify_kernel(K3,A3s);
errs(3) = misclasserr(Y,C3s);
		% test misclass errors
C1se = classify(Xtest,W1s);
errse(1) = misclasserr(Ytest,C1se);
C2se = classify_kernel(K2test,A2s);
errse(2) = misclasserr(Ytest,C2se);
C3se = classify_kernel(K3test,A3s);
errse(3) = misclasserr(Ytest,C3se);

%%%%% unsupervised pca %%%%%
[Y1p,U1p,W1p,objp(1)] = pca(X,k,beta1,1);
[Y2p,B2p,A2p,objp(2)] = pca_kernel(K2,k,beta2,1);
[Y3p,B3p,A3p,objp(3)] = pca_kernel(K3,k,beta3,1);
		% train misclass errors
C1p = classify(X,W1p);
errp(1) = minalignerr(Y,C1p);
C2p = classify_kernel(K2,A2p);
errp(2) = minalignerr(Y,C2p);
C3p = classify_kernel(K3,A3p);
errp(3) = minalignerr(Y,C3p);
		% test misclass errors
C1pe = classify(Xtest,W1p);
errpe(1) = minalignerr(Ytest,C1pe);
C2pe = classify_kernel(K2test,A2p);
errpe(2) = minalignerr(Ytest,C2pe);
C3pe = classify_kernel(K3test,A3p);
errpe(3) = minalignerr(Ytest,C3pe);

%%%%% unsupervised kmeans %%%%%
for r = 1:Reps

	[C1,U1,W1,objk(r,1)] = kmeans(X,k,beta1);
	if objk(r,1) < minobjk(1)	% keep best
		minobjk(1) = objk(r,1); 
		C1k = C1;
		W1k = W1;
	end

	[C2,B2,A2,objk(r,2)] = kmeans_kernel(K2,k,beta2);
	if objk(r,2) < minobjk(2)	% keep best
		minobjk(2) = objk(r,2); 
		C2k = C2;
		A2k = A2;
	end

	[C3,B3,A3,objk(r,3)] = kmeans_kernel(K3,k,beta3);
	if objk(r,3) < minobjk(3)	% keep best
		minobjk(3) = objk(r,3); 
		C3k = C3;
		A3k = A3;
	end

%objk(r,:)	% you can turn this off if the output annoys you

end
		% train misclass errors
errk(1) = minalignerr(C1k,Y);
errk(2) = minalignerr(C2k,Y);
errk(3) = minalignerr(C3k,Y);
		% test misclass errors
C1ke = classify(Xtest,W1k);
errke(1) = minalignerr(Ytest,C1ke);
C2ke = classify_kernel(K2test,A2k);
errke(2) = minalignerr(Ytest,C2ke);
C3ke = classify_kernel(K3test,A3k);
errke(3) = minalignerr(Ytest,C3ke);

%%%%%%%%%% report results %%%%%%%%%%

	% report train objective values
%objk	% all kmeans objectives
meanobjk = mean(objk,1)	% mean objective for kmeans
minobjk	% min objective for kmeans
objp	% pca
objs	% lsq

	% report train misclass errs
errk	% kmeans
errp	% pca
errs	% lsq

	% report test misclass errs
errke	% kmeans
errpe	% pca
errse	% lsq

	% plot training embeddings and clusterings
figure
plt(Y1p,Y,C1k,2);
figure
plt(Y2p,Y,C2k,2);
figure
plt(Y3p,Y,C3k,2);
