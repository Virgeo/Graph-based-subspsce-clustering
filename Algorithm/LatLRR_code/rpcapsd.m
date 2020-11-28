function [Z,E] = rpcapsd(X,lambda,rho,display)
%NSPCODE Summary of this function goes here
%   Detailed explanation goes here
% min |Z|_*+lambda*|E|_1
% s.t., X = Z+E
%       Z is psd
if nargin<4
    display = false;
end
if nargin<3
    rho = 2;
end
if nargin<2
    lambda = 1;
end
tol = 1e-6;
maxIter = 1e6;
[d n] = size(X);
max_mu = 1e10;
mu = 1e-6;
%% Initializing optimization variables
% intialize
E = sparse(d,n);
Z = zeros(d,n);

Y = zeros(d,n);
%% Start main loop
iter = 0;
if display
    disp(['initial,rank=' num2str(rank(Z))]);
end
while iter<maxIter
    iter = iter + 1;
    
    temp = X-E + Y/mu;
    temp = (temp + temp')/2;
    [U,D] = eig(temp);
    U = real(U);
    D = real(diag(D));
    inds = D>1/mu;
    if sum(inds)>0
        D = diag(D(inds)-1/mu);
        Z = U(:,inds)*D*U(:,inds)';
    else
        Z = 0;
    end
    rZ = sum(inds);
    
    temp = X-Z+Y/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
    
    leq = X-Z-E;
    stopC = max(max(abs(leq)));
    if display && (iter==1 || mod(iter,50)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rZ) ',|E|=' num2str(sum(sum(abs(E)))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y = Y + mu*leq;
        mu = min(max_mu,mu*rho);
    end
end