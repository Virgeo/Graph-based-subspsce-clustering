function [X, y] = spheres_gen(c, n, noise)
% each row is a data point

if nargin < 3
    noise = 0.03;
end;

k = sqrt(c);
conv=noise*eye(2);
m = [kron(1:k,ones(1,k)); kron(ones(1,k),1:k)];
X=[];
y=[];
for i=1:c
    X=[X,mvnrnd(m(:,i),conv,n/c)'];
    y = [y;i*ones(n/c,1)];
end

X = X';