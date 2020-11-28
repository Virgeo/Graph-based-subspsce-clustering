function [X, y] = three_ring_dim_gen
% generate three ring data
fea_n = 3;
n1 = 120;
n2 = 220; 
n3 = 260;
n = n1 + n2 + n3;
noise1 = 0.035;
noise2 = 0.13;
curve = 2.5;

% 2-D data
r = 0.2;
t = unifrnd(0,0.8,[1,n1]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n1);
y = r.*cos(curve*pi*t) + noise1*randn(1,n1);
z = 5*noise2*randn(fea_n,n1);
data1 = [x;y;z];

r = 0.7;
curve = 2.5;
t = unifrnd(0,0.8,[1,n2]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n2);
y = r.*cos(curve*pi*t) + noise1*randn(1,n2);
z = 5*noise2*randn(fea_n,n2);
data2 = [x;y;z];

r = 1.2;
curve = 2.5;
t = unifrnd(0,0.8,[1,n3]);
x = r.*sin(curve*pi*t) + noise1*randn(1,n3);
y = r.*cos(curve*pi*t) + noise1*randn(1,n3);
z = 5*noise2*randn(fea_n,n3);
data3 = [x;y;z];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

% for three ring data
X = [data1,data2,data3]';
y = [ones(1,n1),2*ones(1,n2),3*ones(1,n3)];