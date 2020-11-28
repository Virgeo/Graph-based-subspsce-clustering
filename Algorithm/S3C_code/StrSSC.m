% StrSSC.m
% Description: This code is for the Structured Sparse Subspace Clustering.
%
% Ref: 
%   Chun-Guang Li and Ren¨¦ Vidal, "Structured Sparse Subspace Clustering: A Unified Optimization Framework", 
%   In CVPR, pp.277-286, June 7-12, 2015.
%
%   [acc_i, Theta, C, eval_iter] = StrSSC(D, idx, opt, DEBUG)
%
%    Inputs:  D - data matrix, each column as a data point
%                 idx - data points groundtruth label;
%                 opt - the parameters setting:
%                     opt.affine: 0 or 1
%                     opt.outliers: 1 or 0
%                     opt.gamma0 - e.g., 0.1 ( 0.1 to 0.25), the parameter to re-weight with Theta
%                     opt.nu  - set as 1, to make the first run of StrSSC the same as SSC
%                     opt.lambda - it is the lambda for other algorithm and the alpha for SSC's code      
%                     opt.r - the dimension of the target space when applying PCA or random projection
%                     opt.iter_max - eg. 5, 10, to set the maximum iterations of the StrSSC's outer loop       
%                     opt.maxIter - the maximum iteration number of ADMM
%                     opt.SSCrho - the thresholding paramter sometimes is used in SSC (e.g. rho = 0.7 for MotionSegmentation). By
%                     default, it is set as 1
%                 
%    Outputs: 
%                acc_i  - accuracy
%                Theta -  subspace membership matrix, i.e. the structure indicator matrix
%                C  -  sparse representation 
%                eval_iter - record the acc in each iteration for StrSSC
%
%    How to Use:
%
%             %% paramters for StrSSC
%             opt.iter_max =10;
%             opt.gamma0 =0.1; % This is for reweighting the off-diagonal entries in Z
%             opt.nu =1;
%
%             %% paramters for ssc
%             opt.affine = 0; 
%             opt.outliers =1;   
%             opt.lambda =10;
%             opt.r =0;             
%             opt.SSCrho =1;            
%
%             %% paramters for ADMM
%             opt.tol =1e-5;
%             opt.rho=1.1;
%             opt.maxIter =150;
%             opt.mu_max = 1e8;
%             opt.epsilon =1e-3;        
%             %opt.tol =1e-3;
%             %opt.rho =1.1;
%
%
%             [acc_i, Theta_i, C, eval_iter] = StrSSC(D, idx, opt);
%
% Copyright by Chun-Guang Li      
% Date: Jan. 7th, 2014
% Modified: Oct.31. 2014
% Revised by July 28, 2015

function [acc_i, Theta, C, eval_iter] = StrSSC(D, idx, opt, DEBUG)
% 
if nargin < 4
    DEBUG =0;
end
if nargin < 3
    opt.iter_max =10;
    opt.nu =1;
    opt.gamma0 =0.1;
    
    opt.affine = 0; % e.g., 'ssc'
    opt.outliers =1;
    opt.lambda =10; % 0.3 for lrr
    opt.r =0;
    opt.SSCrho =1;
    
    opt.tol =1e-3;
    opt.epsilon =1e-3;
    opt.maxIter =150;
    opt.rho =1.1;
end

%% parameter settings:
% StrSSC specific parameters
iter_max =opt.iter_max; % iter_max =10
gamma0 =opt.gamma0;
nu =opt.nu; % nu=1;

% parameters used in SSC
affine = opt.affine;
outliers =opt.outliers;
nbcluster = max(idx);
alpha =opt.lambda;        
r = opt.r; % r: the dimension of the target space when applying PCA or random projection
Xp = DataProjection(D,r);
rho =opt.SSCrho; %rho = 1 be default;

% parameters used in ADMM
maxIter =opt.maxIter; % in ADMM
%             admmopt.tol =1e-5;
%             admmopt.rho=1.1;
%             admmopt.maxIter =150;
%             admmopt.mu_max = 1e8;
%             admmopt.epsilon =1e-3;    

%% Initialization
C =zeros(size(D,2));
Theta_old =ones(size(D, 2));
%Theta_old =zeros(size(D, 2));
grps =0; 
eval_iter =zeros(1, iter_max);      
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
    
    %% run ADMM to solve the re-weighted SSC problem
    C = ADMM_StrSSC(Xp, outliers, affine, alpha, Theta_old, gamma1, nu,  maxIter, C);

    %% Initialize Z with the previous optimal solution
    CKSym = BuildAdjacency(thrC(C,rho));
    
    %% without warmstart
    % grps = SpectralClustering(CKSym, nbcluster);     
    %% with 'warmstart' which we initialize the next kmeans with the previous clustering result
    % it call our modified kmeans.
    grps = SpectralClustering(CKSym, nbcluster, grps); 
    
    if (nbcluster >10)
        missrate = 1 - compacc(grps, idx'); % it is an approximate way to calculate ACC.
    else
        missrate = Misclassification(grps, idx'); % it sometimes causes out of memory when nbcluster >10 
    end
    disp(['iter = ',num2str(iter),', acc = ', num2str(1 - missrate)]);
    acc_i =1 - missrate;
    eval_iter(iter) = 1 - missrate;
    Theta = form_structure_matrix(grps);     

    if (DEBUG)
        figure;
        image(abs(C)*800);colorbar;
        figure;
        image(abs(CKSym)*800);colorbar;
        figure;
        image(abs(1-Theta)*800);colorbar;
    end

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