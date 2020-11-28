clc;
close all;

folder_now = pwd;  addpath([folder_now, '\funs']);

newdata = 1;
if newdata == 1
    clear;
    num0 = 200;
    [X, n1] = twogaussian_gen(num0, 0.99, 0.0, .1, 5); c = 2;
end;


[W, la, A, evs] = PCAN(X', c);
figure('name','Learned graph by PCAN'); 
imshow(A,[]); colormap jet; colorbar;


figure('name','Projected directions by PCAN, PCA and LPP'); 
set(gca, 'fontsize',15);
%plot(X(:,1),X(:,2),'.k'); hold on;
plot(X(la==1,1),X(la==1,2),'.r','MarkerSize',15); hold on;
plot(X(la==2,1),X(la==2,2),'.b','MarkerSize',15); hold on;
plot(X(la==3,1),X(la==3,2),'.g','MarkerSize',15); hold on;
minx = 1.5*min(X(:,1)); maxx = 1.5*max(X(:,1));
miny = 1.1*min(X(:,2)); maxy = 1.1*max(X(:,2));
if abs(W(1))>abs(W(2))
    h1 = plot([minx, maxx],[W(2)/W(1)*minx, W(2)/W(1)*maxx],'c','LineWidth',2,'MarkerSize',15); hold on;
else
    h1 = plot([W(1)/W(2)*miny, W(1)/W(2)*maxy], [miny, maxy],'c','LineWidth',2,'MarkerSize',15); hold on;
end;
axis equal;


% pca
num = size(X,1);
H = eye(num)-1/num*ones(num);
St = X'*H;
[U, S, V] = svd(St,'econ'); s = diag(S);
W = U(:,1);
if abs(W(1))>abs(W(2))
    h2 = plot([minx, maxx],[W(2)/W(1)*minx, W(2)/W(1)*maxx],'g','LineWidth',2,'MarkerSize',15); hold on;
else
    h2 = plot([W(1)/W(2)*miny, W(1)/W(2)*maxy], [miny, maxy],'g','LineWidth',2,'MarkerSize',15); hold on;
end;
axis equal;



% lpp
H = eye(num)-1/num*ones(num);
St =X'*H*X;
invSt = inv(St);
A = selftuning(X, 10);
L = diag(sum(A,2))-A;
Sl = X'*L*X;
M = invSt*Sl;
[W, temp, ev]=eig1(M, 2, 0, 0);
W = W*diag(1./sqrt(diag(W'*W)));
if abs(W(1))>abs(W(2))
    h3 = plot([minx, maxx],[W(2)/W(1)*minx, W(2)/W(1)*maxx],'m','LineWidth',2,'MarkerSize',15); hold on;
else
    h3 = plot([W(1)/W(2)*miny, W(1)/W(2)*maxy], [miny, maxy],'m','LineWidth',2,'MarkerSize',15); hold on;
end;
axis equal;

legend([h1,h2,h3],'PCAN','PCA','LPP',4);
