%--------------------------------------------------------------------------
% This function normalizes the columns of a given matrix 
% Y: DxN data matrix
% Yn: DxN data matrix whose columns have unit Euclidean norm
% n: N-dimensional vector of the norms of the columns of Y
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [Y,n] = matrixNormalize(Y)

for i = 1:size(Y,2)
    n(i) = norm(Y(:,i));
    Y(:,i) = Y(:,i) ./ n(i);
end