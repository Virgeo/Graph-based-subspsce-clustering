function [X, y] = three_ring_gen(n, noise1, fea_n, noise2)
% generate three ring data, each row is a data

n1 = floor(1/10*n);
n2 = floor(3/10*n);
n3 = floor(6/10*n);

if nargin < 4
    noise2 = 0.1;%0.139;
end;
if nargin < 3
    fea_n = 1;
end;
if nargin < 2
    noise1 = 0.035;
end;

curve = 2.5;

% 2-D data
r = 0.2;
t = unifrnd(0,0.8,[1,n1]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n1);
y = r.*cos(curve*pi*t) + noise1*randn(1,n1);
z = 5*noise2*randn(fea_n,n1);
data1 = [x;y;z];

r = 0.6;
curve = 2.5;
t = unifrnd(0,0.8,[1,n2]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n2);
y = r.*cos(curve*pi*t) + noise1*randn(1,n2);
z = 5*noise2*randn(fea_n,n2);
data2 = [x;y;z];

r = 1;
curve = 2.5;
t = unifrnd(0,0.8,[1,n3]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n3);
y = r.*cos(curve*pi*t) + noise1*randn(1,n3);
z = 5*noise2*randn(fea_n,n3);
data3 = [x;y;z];

X = [data1,data2,data3];
y = [ones(n1,1);2*ones(n2,1);3*ones(n3,1)];
X = X';
if nargin < 3
    X = X(:,1:2);
end;
