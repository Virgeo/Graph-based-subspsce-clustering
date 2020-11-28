% T his routine uses Inexact ALM algorithm to solve the following nuclear-norm optimization problem:
% min |Z|_*+lambda*|E|_2,1
% s.t., X = XZ+E
% 
% reference:  G. Liu, Z. Lin, S. Yan, J. Sun, and Y. Ma, ¡°Robust recovery of
% subspace structures by low-rank representation,¡±IEEE Trans. Pattern Anal.
% Mach.Intell., vol. 35, no. 1, pp. 171¨C184, Jan. 2013.
function Z = lrr(X,lambda)

[m,n] = size(X);
%% set up regularization parameters
mu = 1e-6;
mu_max = 1e10;
rho = 1.1;
epsilon = 1e-4;

%% initialize parameters
Z = zeros(n);
J = zeros(n);
E = zeros(m,n);
Y1 = zeros(m,n);
Y2 = zeros(n);

%% initialize X
% X = matrixNormalize(X);

X1 = X'*X;
%% self-defined function
shk = @(X, tau)(sign(X).*max(abs(X)-tau, 0));


%% iterative procedure
k = 0;
while(1)
    k = k+1;
    % update J
    [U, S, V] = svd(Z+Y2/mu,'econ');
     diagS = diag(S);
     svp = length(find(diagS > 1/mu));
     if svp >= 1
        diagS = diagS(1:svp) - 1/mu; 
     else
         svp = 1;
         diagS = 0;
     end
     J = U(:,1:svp)*diag(diagS)*V(:,1:svp)'; 
     

    % update Z
    T1 = eye(n) + X1;
    T2 = X'*Y1 - Y2;
    Z = T1\(X1 - X'*E + J + T2/mu);
    
    % update E
    Q = X - X*Z + Y1/mu;
    E = solve_l1l2(Q,lambda/mu);
    
    % update Y1 ,Y2
    Y1 = Y1 + mu*(X - X*Z - E);
    Y2 = Y2 + mu*(Z - J);
    % update  mu
    mu = min(rho*mu,mu_max);
    
    % check convergence
    del(1) = norm(X-X*Z-E,inf);
    del(2) = norm(Z-J,inf);
    if max(del) < epsilon
        break
    else
     fprintf('Iter %3d: del_Z=%5.3e   del_J=%5.3e  \n', k, del(1), del(2));
    
   

     end
end

