% run_StrSSC_Faces.m
% 
% This code is modified from Esan's SSC code.
%
clc;
clear all
close all
%addpath('../lrr');
addpath('./CodefromSSC');
% addpath('../SSC_ADMM_v1.1');

 load YaleBCrop025.mat
load 'C:\Users\csjunxu\Desktop\SC\Datasets\YaleB_Crop.mat'  
results_fn =['StrSSC_Faces_tuned_gamma0_results',datestr(now,30),'.mat'];
alpha = 20;
gamma0 =0.1; % 0.1 for Face
%lambda =0.15; % for LRR

nSet = [2 3 5 8 10];
for i = 1:length(nSet)    
    n = nSet(i);
    idx = Ind{n};   
    for j = 1:size(idx,1)
        X = [];
        for p = 1:n
            X = [X Y(:,:,idx(j,p))];
        end
        [D,N] = size(X);
       
        tic
        disp(['For number of clusters n =  ', num2str(n),':  There are totally ', num2str(size(idx,1)), ' cases. ']);
        %% 1. SSC
        r = 0; affine = false; outlier = true; rho = 1;
        %[missrate1,C] = SSC(X,r,affine,alpha,outlier,rho,s{n});
        missrate1 =0; 
        missrateTot1{n}(j) = missrate1;
        
        disp(['------------------------- - - n =  ', num2str(n),', j = ', num2str(j), ' of  ', num2str(size(idx,1))]);
        disp(['* * * SSC missrate 1: ', num2str( missrate1)]);

         %% 2. StrSSC 
        % paramters for standard SSC
        opt.affine =0;
        opt.outliers =1;        
        opt.lambda =alpha;
        opt.r =0;  % the dimension of the target space when applying PCA or random projection
        opt.SSCrho=1;
        
        % paramters for StrSSC
        opt.iter_max =10; %  iter_max is for loop in StrLRSCE
        opt.nu =1;
        opt.gamma0 =gamma0;% This is for reweighting the off-diagonal entries in Z

        
        %% paramters for ADMM
        %opt.tol =1e-5;
        %opt.rho=1.1;
        opt.maxIter =150;
        %opt.mu_max = 1e8;
        %opt.epsilon =1e-3;        
        %opt.tol =1e-3;
        %opt.rho =1.1;
        
        [missrate2] = StrSSC(X, s{n}', opt);            
        missrateTot2{n}(j) = 1 - missrate2;
        disp(['* * * StrSSC missrate 2: ', num2str(1-missrate2)]);    
        disp(['------------------------']);            
            
        t2 =toc;
        disp(['Esclapsed time:  ', num2str( t2 *                       j )]);
        disp(['Reminding time: ', num2str( t2 * (size(idx,1) - j )),'(s) ,  about  ', num2str( t2 * (size(idx,1) - j )/3600), ' hours.']);  

        save(results_fn, 'missrateTot1', 'missrateTot2',  'alpha', 'gamma0','opt');
    end
    avgmissrate1(n) = mean(missrateTot1{n});
    medmissrate1(n) = median(missrateTot1{n});
    avgmissrate2(n) = mean(missrateTot2{n});
    medmissrate2(n) = median(missrateTot2{n});

    save(results_fn, 'missrateTot1', 'avgmissrate1', 'medmissrate1', 'missrateTot2', 'avgmissrate2', 'medmissrate2', 'nSet', 'alpha', 'gamma0','opt');
end