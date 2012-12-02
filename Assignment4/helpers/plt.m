function [] = plt(X,Y,C,d)
% plot first d dimensions of X
% showing the clusterings given by C
%	Y - colors
%	C - symbols

t = size(X,1);
I = randperm(t);

colors = 'grbcmykgrb';
symbols = '.ox+*sdv^p';

%clf
%axis square
%hold
%plot3(X(1,1),X(1,2),X(1,3))
for i = 1:t
	if d == 2
		plot(X(I(i),1),X(I(i),2),[colors(find(Y(I(i),:))) symbols(find(C(I(i),:)))]);
	elseif d == 3
		st = [colors(find(Y(I(i),:))) symbols(find(C(I(i),:)))];
		plot3(X(I(i),1),X(I(i),2),X(I(i),3),st);
           
	end
	hold on
end

