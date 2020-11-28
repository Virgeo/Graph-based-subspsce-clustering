function [X, n1] = twogaussian_gen(num, interval1, interval2, var1, var2)

n1 = floor(num/2);
m = [-interval1,interval2];
C = [var1,0;0,var2];
x1 = mvnrnd(m,C,n1);

m = [interval1,-interval2];
x2 = mvnrnd(m,C,n1);

X = [x1;x2];