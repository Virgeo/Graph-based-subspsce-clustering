% A demo code corresponding to the reference:
%"Robust Subspace Segmentation by Simultaneously Learning Data 
% Representations and Their Affinity Matrix" by Xiaojie Guo, IJCAI 2015
% If any problems, please contact Xiaojie via xj.max.guo@gmail.com



close all; clear all;clc
addpath(genpath('./'));

%% Data Preparation 
load('USPS.mat');
row = true;
nCluster = 10;           % number of subspace
%nSam = 64 ;             % sample number from each class
fea = fea';
gnd = gnd';
feaT = [];
gndT = [];
for i = 1 : nCluster
    indT = find(gnd==i);
    indT = indT(1:100);
    feaT = [feaT, fea(:,indT(:))];
    gndT = [gndT, gnd(indT(:))]; 
end

fea = feaT;
gnd = gndT;

for i = 1 : size(fea,2)
   fea(:,i) = fea(:,i) /norm(fea(:,i)) ; 
end

%% Parameters

rep = 10;         % average over rep runs
t= .8;            % \hat{\lambda} ![YaleB: t = 0.1; USPS: 0.8; UMIST: 1.0]!

% The performance would be further improved by tuning [beta, gamma and eta] 
% but, for the simplicity of model, we set them the same in the paper.
coef.beta=t;
coef.gamma=t;
coef.eta=t;
coef.k = 3;       % k nearest neighbors
coef.cluNb = nCluster;

%% Do the job

[R,A,E] = SubClu_SimR(fea,coef);

%% Accuracy

disp(['Accuracy using A only: ', num2str(computeACC(A,gnd,rep))])
disp(['Accuracy using R only: ', num2str(computeACC(R,gnd,rep))])
disp(['Accuracy using A + R:  ', num2str(computeACC(A.*R,gnd,rep))])

