function C = admm_S3C(X,para,Theta_old)
lambda = para.lambda;
alpha = para.alpha;

[m, n] = size(X);
%% set up regularization parameters
epsilon1 = 1e-6;
mu = 1e-2;
mu_max = 1e10;
rho = 1.1;
maxIter = 150; 

%% initialization
A = zeros(n);
C = zeros(n);
E = zeros(m, n);
Y1 = zeros(m, n);
Y2 = zeros(n);
X1 = X'*X;


%% self-defined function
 shk = @(X, tau)(sign(X).*max(abs(X)-tau, 0));
%% iterative procedure
iter =0; 
while(iter < maxIter)
    iter = iter +1;  
    %% update representation C
    U = A + Y2/mu;
    C = shk(U,(1+alpha*Theta_old)/mu);
    C = C - diag(diag(C));
    
    %% update A
    T1 = X - E - Y1/mu;
    A = (X1 + eye(n))\(X'*T1 + C - Y2/mu);
    
    %% update E
    V = X - X*A + Y1/mu;
    E = shk(V,lambda/mu);
    
    %% update Lagrange multipliers
    Y1 = Y1 + mu*(X - X*A -E);
    Y2 = Y2 + mu*(A - C);
    
    mu = min(mu_max, rho*mu);
    
	%% Check convergence
    del = norm(X - X*A -E, 'inf');
    if(del < epsilon1)
        break
    else
          fprintf('Iter %3d: del=%5.3e \n', ...
            iter, del);
    end
end
