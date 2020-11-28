% min_{A>=0, A*1=1, W'*St*W=I, F'*F=I}  \sum_ij aij*||W'*xi-W'*xj||^2 + r*||A||^2 + 2*lambda*trace(F'*L*F)
% written by Feiping Nie on 2/9/2014
function [W, y, A, evs] = PCAN(X, c, d, k, r, islocal)
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% d: projected dimension
% k: number of neighbors to determine the initial graph, and the parameter r if r<=0
% r: paremeter, which could be set bo a large enough value. If r<0, then it is determined by algorithm with k
% islocal: 
%           1: only update the similarities of the k neighbor pairs, the neighbor pairs are determined by the distances in the original space 
%           0: update all the similarities
% W: dim*d projection matrix
% y: num*1 cluster indicator vector
% A: num*num learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations

% For more details, please see:
% Feiping Nie, Xiaoqian Wang, Heng Huang. 
% Clustering and Projected Clustering with Adaptive Neighbors.  
% The 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), New York, USA, 2014.



NITER = 30;
num = size(X,2);  
if nargin < 6
    islocal = 0;
end;
if nargin < 5
    r = -1;
end;
if nargin < 4
    k = 15;
end;
if nargin < 3
    d = c-1;
end;

distX = L2_distance_1(X,X);
%distX = sqrt(distX);
[distX1, idx] = sort(distX,2);
A = zeros(num);
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;

if r <= 0
    r = mean(rr);
end;
lambda = r;

A0 = (A+A')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;
[F, temp, evs]=eig1(L0, c, 0);

H = eye(num)-1/num*ones(num);
St = X*H*X';
invSt = inv(St);
M = (X*L0*X');
W = eig1(M, d, 0, 0);

for iter = 1:NITER
    distf = L2_distance_1(F',F');
    distx = L2_distance_1(W'*X,W'*X);
    if iter>5
        [temp, idx] = sort(distx,2);
    end;
    A = zeros(num);
    for i=1:num
        if islocal == 1
            idxa0 = idx(i,2:k+1);
        else
            idxa0 = 1:num;
        end;
        dfi = distf(i,idxa0);
        dxi = distx(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;

    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    
    M = invSt*(X*L*X');
    W = eig1(M, d, 0, 0);
    W = W*diag(1./sqrt(diag(W'*W)));
    
    F_old = F;
    [F, temp, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 0.000000001
        lambda = 2*lambda;
    elseif fn2 < 0.00000000001
        lambda = lambda/2;  F = F_old;
    else
        break;
    end;

end;

%[labv, tem, y] = unique(round(0.1*round(1000*F)),'rows');
[clusternum, y]=graphconncomp(sparse(A)); y = y';
if clusternum ~= c
    sprintf('Can not find the correct cluster number: %d', c)
end;


