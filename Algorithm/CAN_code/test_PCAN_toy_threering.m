clc;
close all;

folder_now = pwd;  addpath([folder_now, '\funs']);

newdata = 1;
datatype = 1; % 1: three-Gaussian data, 2: three-ring data

if newdata == 1
    clearvars -except datatype;
    
    if datatype == 1
        [X, y, n1, n2, n3] = threegaussian_dim_gen(300, 0.05, 10, 0.5); c = 3;
    else
        [X, y] = three_ring_dim_gen; c = 3;
    end;
end;


[W, la, A, evs] = PCAN(X', c);

X1 = X*W;
figure('name','Learned subspace and clustering by PCAN'); 
plot(X1(:,1),X1(:,2),'.k'); hold on;
cm = colormap(jet(c));
for i = 1:c
    plot(X1(la==i,1),X1(la==i,2),'.', 'color', cm(i,:)); hold on;
end;
axis equal;


% pca
num = size(X,1);
H = eye(num)-1/num*ones(num);
St = X'*H;
[U, S, V] = svd(St,'econ'); s = diag(S);
X2 = X*U(:,1:2);
figure('name','Learned subspace by PCA'); 
plot(X2(:,1),X2(:,2),'.k'); hold on;
for i = 1:c
    plot(X2(y==i,1),X2(y==i,2),'.', 'color', cm(i,:)); hold on;
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
X3 = X*W;
figure('name','Learned subspace by LPP'); 
plot(X3(:,1),X3(:,2),'.k'); hold on;
for i = 1:c
    plot(X3(y==i,1),X3(y==i,2),'.', 'color', cm(i,:)); hold on;
end;
axis equal;
