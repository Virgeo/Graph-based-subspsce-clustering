function [x, f] = EProjSimplexdiag(d, u)

%
%% Problem
%
%  min  1/2*x'*U*x-x'*d
%  s.t. x>=0, 1'x=1
%

lambda = min(u-d);
f = 1;
count=1;
while abs(f) > 10^-8
    v1 = 1./u*lambda+d./u;
    posidx = v1>0;
    g = sum(1./u(posidx));
    f = sum(v1(posidx))-1;
    lambda = lambda - f/g;
    
    if count > 1000
        break;
    end;
    count=count+1;
end;
v1 = 1./u*lambda+d./u;
x = max(v1,0);
