% Toy experiment on the two-moon synthetic data.
% Feiping Nie, Xiaoqian Wang, Michael I. Jordan, Heng Huang.
% The Constrained Laplacian Rank Algorithm for Graph-Based Clustering.
% The Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16)

clc;  close all;
currentFolder = pwd;
addpath(genpath(currentFolder));

newdata = 1;
if newdata == 1
    clear
    %% Data Generalization
    num0 = 100;
    X = twomoon_gen(num0);
    y0 = [ones(num0,1);2*ones(num0,1)];
    c = 2;
end;

A0 = constructW_PKN(X', 10, 0);
A = A0;
A = (A+A')/2;

la = y0;
% The original data
MS = 18;
figure; 
plot(X(:,1),X(:,2),'.k', 'MarkerSize', MS); hold on;
plot(X(la==1,1),X(la==1,2),'.b', 'MarkerSize', MS); hold on;
plot(X(la==2,1),X(la==2,2),'.r', 'MarkerSize', MS); hold on;
axis equal;

% Probabilistic neighbors, line width denotes similarity
figure; 
plot(X(:,1),X(:,2),'.k', 'MarkerSize', MS); hold on;
plot(X(la==1,1),X(la==1,2),'.b', 'MarkerSize', MS); hold on;
plot(X(la==2,1),X(la==2,2),'.r', 'MarkerSize', MS); hold on;
nn = 2*num0;
for ii = 1 : nn;
    for jj = 1 : ii
        weight = A(ii, jj);
        if weight > 0
            plot([X(ii, 1), X(jj, 1)], [X(ii, 2), X(jj, 2)], '-g', 'LineWidth', 15*weight), hold on;
        end
    end;
end;
axis equal;

[y, S, evs, cs] = CLR(A0, c);
A = (S+S')/2;

% Adaptive neighbors with L2, line width denotes similarity
figure; 
plot(X(:,1),X(:,2),'.k', 'MarkerSize', MS); hold on;
plot(X(la==1,1),X(la==1,2),'.b', 'MarkerSize', MS); hold on;
plot(X(la==2,1),X(la==2,2),'.r', 'MarkerSize', MS); hold on;
nn = 2*num0;
for ii = 1 : nn;
    for jj = 1 : ii
        weight = A(ii, jj);
        if weight > 0
            plot([X(ii, 1), X(jj, 1)], [X(ii, 2), X(jj, 2)], '-g', 'LineWidth', 15*weight), hold on;
        end
    end;
end;
axis equal;
