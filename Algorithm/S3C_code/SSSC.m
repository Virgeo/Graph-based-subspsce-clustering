function [C,Theta] = SSSC(X,para)

alpha = para.alpha;
gamma0 = para.gamma0;
nCluster = para.nCluster;

%% set up regularization parameters


maxIter = 150; 
iter_max =5;
nu =1;
% gamma0 =0.1;
affine = 0; % e.g., 'ssc'
outliers =1;
% alpha =10; % 0.3 for lrr

%% initialization
[m,n] = size(X);
C = zeros(n);

Theta_old =ones(n);
%% iterative procedure
iter =0; 
while (iter < iter_max)        
    iter = iter +1;    
    gamma1 =gamma0;
    if (iter <= 1)
        %% This is the standard SSC when iter <=1
        nu =1; 
    else
        %% This is for re-weighted SSC
        nu = nu * 1.2;%1.1, 1.2, 1.5,      
    end
    C = ADMM_StrSSC(X, alpha,outliers, affine,  Theta_old, gamma1, nu,  maxIter, C);
    W = (abs(C) + abs(C'))/2;
    grps = SpectralClustering(W,nCluster);
    Theta = form_structure_matrix(grps);     
     %% Checking stop criterion
    tmp =Theta - Theta_old;
    if (max(abs(tmp(:))) < 0.5)
        break; % if Q didn't change, stop the iterations.
    end
    Theta_old =Theta;
end
function M = form_structure_matrix(idx,n)
if nargin<2
    n =size(idx,2);
end
M =zeros(n);
id =unique(idx);
for i =1:length(id)
    idx_i =find(idx == id(i));
    M(idx_i,idx_i)=ones(size(idx_i,2));
end  