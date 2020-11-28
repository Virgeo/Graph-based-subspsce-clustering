clc; close all;
currentFolder = pwd;
addpath(genpath(currentFolder));

newdata = 1;
if newdata == 1
    clear;
    %% Data Generalization
    n = 100; % Total number of points
    c = 4; % Number of clusters
    n1 = n/c;
    noisePortion = 0.85;
    noise = noisePortion*rand(n) + 10*blkdiag(ones(n1),ones(n1),ones(n1),ones(n1));
    noise(noise>1) = 0;
    c1=1;A=blkdiag(c1*rand(n1),c1*rand(n1),c1*rand(n1),c1*rand(n1)) + noise; 

    % Randomly set 20 noise points to 1
    A(1,100)=1; A(51,16)=1; A(76,12)=1; A(1,87)=1; A(3,95)=1; A(30,1)=1; A(77,9)=1; A(50,22)=1; A(8,88)=1; A(21,93)=1; A(45,8)=1; A(27,91)=1;
    A(17,97)=1; A(53,11)=1; A(34,4)=1; A(28,67)=1; A(24,71)=1; A(2,78)=1; A(75,18)=1; A(17,99)=1;

    A = A - diag(diag(A));
    A0 = A;
    A = (A+A')/2;
else
    A = A0;
end;

y0 = [ones(n1,1);2*ones(n1,1);3*ones(n1,1);4*ones(n1,1)];

figure; imshow(A,[]); colormap jet; colorbar;
set(gcf,'outerposition',get(0,'screensize'));

isrobust = 0;
[y, S, evs, cs] = CLR(A0, c, isrobust);
result_CLR0 = ClusteringMeasure(y0, y)
%n = histc(y,unique(y))'

figure; imshow(S,[]); colormap jet; colorbar;
set(gcf,'outerposition',get(0,'screensize'));

isrobust = 1;
[y, S, evs, cs] = CLR(A0, c, isrobust);
result_CLR1 = ClusteringMeasure(y0, y)
%n = histc(y,unique(y))'

figure; imshow(S,[]); colormap jet; colorbar;
set(gcf,'outerposition',get(0,'screensize'));



% RCut & NCut
D = diag(sum(A));
nRepeat = 100;
Ini = zeros(n, nRepeat);
for jj = 1 : nRepeat
    Ini(:, jj) = randsrc(n, 1, 1:c);
end;

% RCut
fprintf('Ratio Cut\n');
[Fg, tmpD] = eig1(full(D-A), c, 0, 1);
Fg = Fg./repmat(sqrt(sum(Fg.^2,2)),1,c);  %optional
y = tuneKmeans(Fg, Ini);
result_RCut = ClusteringMeasure(y0, y)

% NCut
fprintf('Normalized Cut\n');
Dd = diag(D);
Dn = spdiags(sqrt(1./Dd),0,n,n);
An = Dn*A*Dn;
An = (An+An')/2;
[Fng, D] = eig1(full(An), c, 1, 1);
Fng = Fng./repmat(sqrt(sum(Fng.^2,2)),1,c);  %optional
y = tuneKmeans(Fng, Ini);
result_NCut = ClusteringMeasure(y0, y)


