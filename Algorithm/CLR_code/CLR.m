% min_{S>=0, S*1=1, F'*F=I}  ||S - A||^2 + r*||S||^2 + 2*lambda*trace(F'*L*F)
% or
% min_{S>=0, S*1=1, F'*F=I}  ||S - A||_1 + r*||S||^2 + 2*lambda*trace(F'*L*F)
function [y, S, evs, cs] = CLR(A0, c, isrobust, islocal)
% A0: the given affinity matrix
% c: cluster number
% isrobust: solving the second (L1 based) problem if isrobust=1
% islocal: only update the similarities of neighbors if islocal=1
% y: the final clustering result, cluster indicator vector
% S: learned symmetric similarity matrix
% evs: eigenvalues of learned graph Laplacian in the iterations
% cs: suggested cluster numbers, effective only when the cluster structure is clear

% Ref:
% Feiping Nie, Xiaoqian Wang, Michael I. Jordan, Heng Huang.
% The Constrained Laplacian Rank Algorithm for Graph-Based Clustering.
% The 30th Conference on Artificial Intelligence (\textbf{AAAI}), Phoenix, USA, 2016.


NITER = 30;
zr = 10e-11;
lambda = 0.1;
r = 0;

if nargin < 4
    islocal = 1;
end;
if nargin < 3
    isrobust = 0;
end;

A0 = A0-diag(diag(A0));
num = size(A0,1);
A10 = (A0+A0')/2;
D10 = diag(sum(A10));
L0 = D10 - A10;

% automatically determine the cluster number
[F0, ~, evs]=eig1(L0, num, 0);
a = abs(evs); a(a<zr)=eps; ad=diff(a);
ad1 = ad./a(2:end); 
ad1(ad1>0.85)=1; ad1 = ad1+eps*(1:num-1)'; ad1(1)=0; ad1 = ad1(1:floor(0.9*end));
[te, cs] = sort(ad1,'descend');
% sprintf('Suggested cluster number is: %d, %d, %d, %d, %d', cs(1),cs(2),cs(3),cs(4),cs(5))
if nargin == 1
    c = cs(1);
end;
F = F0(:,1:c);
if sum(evs(1:c+1)) < zr
    error('The original graph has more than %d connected component', c);
end;
if sum(evs(1:c)) < zr
    [clusternum, y]=graphconncomp(sparse(A10)); y = y';
    S = A0;
    return;
end;

for i=1:num
    a0 = A0(i,:);
    if islocal == 1
        idxa0 = find(a0>0);
    else
        idxa0 = 1:num;
    end;
    u{i} = ones(1,length(idxa0));
end;


for iter = 1:NITER
    dist = L2_distance_1(F',F');
    S = zeros(num);
    for i=1:num
        a0 = A0(i,:);
        if islocal == 1
            idxa0 = find(a0>0);
        else
            idxa0 = 1:num;
        end;
        ai = a0(idxa0);
        di = dist(i,idxa0);
        if isrobust == 1
            for ii = 1:1
                ad = u{i}.*ai-lambda*di/2;
                si = EProjSimplexdiag(ad, u{i}+r*ones(1,length(idxa0)));
                u{i} = 1./(2*sqrt((si-ai).^2+eps));
            end;
            S(i,idxa0) = si;
        else
            ad = ai-0.5*lambda*di; S(i,idxa0) = EProjSimplex_new(ad);
        end;
    end;
    A = S;
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    F_old = F;
    [F, ~, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;

    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
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


