function [X, y, n1, n2, n3] = threegaussian_dim_gen(num, var1, fea_n, noise)

n1 = floor(num/3);
m = [-1,0];
C = [var1,0;0,var1];
x1 = mvnrnd(m,C,n1);

m = [1,0];
x2 = mvnrnd(m,C,n1);

m = [0,1];
x3 = mvnrnd(m,C,n1);

z = 5*noise*randn(3*n1,fea_n);

X = [[x1;x2;x3],z];

n2 = n1;
n3 = n1;
y = [ones(n1,1);2*ones(n2,1);3*ones(n3,1)];