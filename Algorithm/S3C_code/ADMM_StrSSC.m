% ADMM_StrSSC.m
% Decsription: This code is, modified from Esan's SSC code, for solving the Structured SSC which is an iterated re-weighted scheme for  the subspace
% segmentation.
%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the sparse representation 
% of each data point in terms of the rest of the points
% Y: DxN data matrix
% affine: if true then enforce the affine constraint
% thr1: stopping threshold for the coefficient error ||Z-C||
% thr2: stopping threshold for the linear system error ||Y-YZ||
% maxIter: maximum number of iterations of ADMM
% C2: NxN sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------
%
% Modified by Chunguang Li
% Date: Jan 21, 2014.

function C2 = ADMM_StrSSC(Y, alpha, affine,outliers, Theta, gamma0, nu,  maxIter, Z, thr)
if (nargin < 2)
    % default regularizarion parameters
    alpha = 800;
    
end
if (nargin < 3)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 4)
    % default subspaces are linear
    outliers = false; 
end
if (nargin < 5)
    % default regularizarion parameters
    Theta = zeros(size(Y, 2));
end
if (nargin < 6)
    gamma0 =0.1;
end
if (nargin < 7)
    nu =1;
end
if (nargin < 8)
    % default maximum number of iterations of ADMM
    maxIter = 200; 
end
if (nargin < 9)
    % default maximum number of iterations of ADMM
    Z = zeros(size(Y, 2));
end
if (nargin < 10)
    % default coefficient error threshold to stop ADMM
    % default linear system error threshold to stop ADMM
    thr = 2*10^-4; 
end

if (~outliers)
    
    %% ADMM_Lasso_weighted
    if (length(alpha) == 1)
        alpha1 = alpha(1);
        alpha2 = alpha(1);
    elseif (length(alpha) == 2)
        alpha1 = alpha(1);
        alpha2 = alpha(2);
    end

    if (length(thr) == 1)
        thr1 = thr(1);
        thr2 = thr(1);
    elseif (length(thr) == 2)
        thr1 = thr(1);
        thr2 = thr(2);
    end

    N = size(Y,2);

    % setting penalty parameters for the ADMM
    mu1 = alpha1 * 1/computeLambda_mat(Y);
    mu2 = alpha2 * 1;
    if (~affine)
        % initialization
        A = inv(mu1*(Y'*Y)+mu2*eye(N));
        %C1 = zeros(N,N);
        C1 =Z;
        Lambda2 = zeros(N,N);
        err1 = 10*thr1; err2 = 10*thr2;
        i = 1;
        % ADMM iterations
        while ( err1(i) > thr1 && i < maxIter )
            % updating Z
            Z = A * (mu1*(Y'*Y)+mu2*(C1-Lambda2/mu2));
            Z = Z - diag(diag(Z));
            % updating C
            % C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2);        
            C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*(ones(N)./nu+ nu*gamma0 * (1-Theta)))) .* sign(Z+Lambda2/mu2);
            C2 = C2 - diag(diag(C2));
            % updating Lagrange multipliers
            Lambda2 = Lambda2 + mu2 * (Z - C2);
            % computing errors
            err1(i+1) = errorCoef(Z,C2);
            err2(i+1) = errorLinSys(Y,Z);
            %
            C1 = C2;
            i = i + 1;
        end
        fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end),err2(end),i);
    else
        % initialization
        A = inv(mu1*(Y'*Y)+mu2*eye(N)+mu2*ones(N,N));
        %C1 = zeros(N,N);
        C1 =Z;
        Lambda2 = zeros(N,N);
        lambda3 = zeros(1,N);
        err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr1;
        i = 1;
        % ADMM iterations
        while ( (err1(i) > thr1 || err3(i) > thr1) && i < maxIter )
            % updating Z
            Z = A * (mu1*(Y'*Y)+mu2*(C1-Lambda2/mu2)+mu2*ones(N,1)*(ones(1,N)-lambda3/mu2));
            Z = Z - diag(diag(Z));
            % updating C
         %C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2* ones(N)))                                                   .* sign(Z+Lambda2/mu2);
         %C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*(ones(N)     +        nu*gamma0 *(1-Theta)))) .* sign(Z+Lambda2/mu2);
            C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*(ones(N)./nu+ nu* gamma0 *(1-Theta)))) .* sign(Z+Lambda2/mu2);
            C2 = C2 - diag(diag(C2));
            % updating Lagrange multipliers
            Lambda2 = Lambda2 + mu2 * (Z - C2);
            lambda3 = lambda3 + mu2 * (ones(1,N)*Z - ones(1,N));
            % computing errors
            err1(i+1) = errorCoef(Z,C2);
            err2(i+1) = errorLinSys(Y,Z);
            err3(i+1) = errorCoef(ones(1,N)*Z,ones(1,N));
            %
            C1 = C2;
            i = i + 1;
        end
        fprintf('err1: %2.4f, err2: %2.4f, err3: %2.4f, iter: %3.0f \n',err1(end),err2(end),err3(end),i);
    end
    
else 
    
    %% ADMM_SSC_outliers_weighted_tune1 
    if (length(alpha) == 1)
        alpha1 = alpha(1);
        alpha2 = alpha(1);
        alpha3 = alpha(1);
    elseif (length(alpha) == 2)
        alpha1 = alpha(1);
        alpha2 = alpha(2);
        alpha3 = alpha(2);
    elseif (length(alpha) == 3)
        alpha1 = alpha(1);
        alpha2 = alpha(2);
        alpha3 = alpha(3);
    end

    if (length(thr) == 1)
        thr1 = thr(1);
        thr2 = thr(1);
    elseif (length(thr) == 2)
        thr1 = thr(1);
        thr2 = thr(2);
    end

    [D,N] = size(Y);

    gamma = alpha3 / norm(Y,1);
    P = [Y eye(D)/gamma];

    % setting penalty parameters for the ADMM
    mu1 = alpha1 * 1/computeLambda_mat(Y, Y);
    mu2 = alpha2 * 1;
    %mu3 =alpha1 * 1/computeLambda_mat(Y, eye(D)/gamma);

    if (~affine)
        % initialization
        A = inv(mu1*(P'*P)+mu2*eye(N+D));
        %C1 = zeros(N+D,N);
        C1 =[Z; zeros(D, N)];
        Lambda1 = zeros(D,N);
        Lambda2 = zeros(N+D,N);
        err1 = 10*thr1; err2 = 10*thr2;
        i = 1;
        % ADMM iterations
        while ( (err1(i) > thr1 || err2(i) > thr2) && i < maxIter )
            % updating Z
            Z = A * (mu1*P'*(Y+Lambda1/mu1)+mu2*(C1-Lambda2/mu2));
            Z(1:N,:) = Z(1:N,:) - diag(diag(Z(1:N,:)));
            % updating C
         %C2                      = max(0,(abs(Z                     +Lambda2          /mu2) - 1/mu2* ones(N+D,N)))                                              .* sign(Z           +Lambda2         /mu2);        
            C2(N+1:N+D,:)  = max(0,(abs(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2) - 1/mu2* ones(D,     N)))                                                   .* sign(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2);        
            %C2(1:N,:)            = max(0,(abs(Z(1:N,:)          +Lambda2(1:N,:)            /mu2) - 1/mu2*(ones(N,     N) + nu*gamma0 *(1-Theta)))) .* sign(Z(1:N,:)          +Lambda2(1:N,:)           /mu2);
            C2(1:N,:)           = max(0,(abs(Z(1:N,:)          +Lambda2(1:N,:)            /mu2) - 1/mu2*(ones(N,      N)./nu+ nu*gamma0 *(1-Theta)))) .* sign(Z(1:N,:)+Lambda2(1:N,:)/mu2);
            C2(1:N,:) = C2(1:N,:) - diag(diag(C2(1:N,:)));
            % updating Lagrange multipliers
            Lambda1 = Lambda1 + mu1 * (Y - P * Z);
            Lambda2 = Lambda2 + mu2 * (Z - C2);
            % computing errors
            err1(i+1) = errorCoef(Z,C2);
            err2(i+1) = errorLinSys(P,Z);
            %
            C1 = C2;
            i = i + 1;
        end
        fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end),err2(end),i);
    else
        % initialization
        delta = [ones(N,1);zeros(D,1)];
        A = inv(mu1*(P'*P)+mu2*eye(N+D)+mu2*(delta*delta'));
        %C1 = zeros(N+D,N);
        C1 =[Z; zeros(D, N)];
        Lambda1 = zeros(D,N);
        Lambda2 = zeros(N+D,N);
        lambda3 = zeros(1,N);
        err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr1;
        i = 1;
        % ADMM iterations
        while ( (err1(i) > thr1 || err2(i) > thr2 || err3(i) > thr1) && i < maxIter )
            % updating Z
            Z = A * (mu1*P'*(Y+Lambda1/mu1)+mu2*(C1-Lambda2/mu2)+mu2*delta*(ones(1,N)-lambda3/mu2));
            Z(1:N,:) = Z(1:N,:) - diag(diag(Z(1:N,:)));
            % updating C
         %C2                      = max(0,(abs(Z                     +Lambda2                     /mu2)  - 1/mu2* ones(N+D,N)))                              .* sign(Z           +Lambda2/mu2);
            C2(N+1:N+D,:)  = max(0,(abs(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2) - 1/mu2* ones(D, N) ))                                                   .* sign(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2);        
            C2(1:N,:)            = max(0,(abs(Z(1:N,:)           +Lambda2(1:N,:)          /mu2)  - 1/mu2*(ones(N    )./nu + nu*gamma0* (1-Theta)))) .* sign(Z(1:N,:)           +Lambda2(1:N,:)           /mu2);
         %C2(1:N,:)           = max(0,(abs(Z(1:N,:)           +Lambda2(1:N,:)          /mu2)  - 1/mu2*(ones(N)            + gamma0     * (1-Theta)))) .* sign(Z(1:N,:)           +Lambda2(1:N,:)           /mu2);
            C2(1:N,:) = C2(1:N,:) - diag(diag(C2(1:N,:)));
            % updating Lagrange multipliers
            Lambda1 = Lambda1 + mu1 * (Y - P * Z);
            Lambda2 = Lambda2 + mu2 * (Z - C2);
            lambda3 = lambda3 + mu2 * (delta'*Z - ones(1,N));
            % computing errors
            err1(i+1) = errorCoef(Z,C2);
            err2(i+1) = errorLinSys(P,Z);
            err3(i+1) = errorCoef(delta'*Z,ones(1,N));
            %
            C1 = C2;
            i = i + 1;
        end
        fprintf('err1: %2.4f, err2: %2.4f, err3: %2.4f, iter: %3.0f \n',err1(end),err2(end),err3(end),i);
    end

end
C2 =C2(1:N,:);