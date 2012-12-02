function [] = pltletter(X,Y,C,d)
% plot first d dimensions of X
% showing the clusterings given by C
%	Y - symbols
%	C - colors

t = size(X,1);
mins = min(X);
maxs = max(X);
colors  = 'ymcrgbk';
c8 = [0.4, 0.4, 0.4]; % extra colors
c9 = [1, 0.4, 0.6];
c10 = [0.5, 0.5, 0.5];

symbols = 'aegcilnoru';

%colors are clusters
%symbols are original class
clf
if d == 2
	axis([mins(1) maxs(1) mins(2) maxs(2)]);
elseif d == 3
	axis([mins(1) maxs(1) mins(2) maxs(2) mins(3) maxs(3)]);
end
axis square
hold

for i = 1:t

	Cf = find(C(i,:));
	if Cf == 8
		col = c8;
	elseif Cf == 9
		col = c9;
	elseif Cf == 10
		col = c10;
	else
		col = colors(find(C(i,:)));  
	end

	if d == 2
		text(X(i,1),X(i,2),symbols(find(Y(i,:))),'color',col);
	elseif d ==3
		text(X(i,1),X(i,2),X(i,3),symbols(find(Y(i,:))),'color',col);
	end

end

hold
