% self-tuning
function [A An] = selftuning(X_total, k)
% each row is a data point

AA = L2_distance_1(X_total',X_total');
AA(find(AA<0)) = 0;
clear X_total;
AA = sqrt(AA);
n = size(AA, 1);
[dumb idx] = sort(AA, 2); % sort each row
clear dumb;
A = zeros(n);
for i = 1:n
    A(i, idx(i,2:k+1)) = AA(i, idx(i,2:k+1)) + eps;
end;
clear AA;
A = max(A, A');
%A = (A+A')/2;
clear idx;

% Find the count of nonzero for each column
col_count = sum(A~=0, 1)';
col_sum = sum(A, 1)';
col_mean = col_sum ./ col_count;
[x y val] = find(A);
A = sparse(x, y, -val.*val./col_mean(x)./col_mean(y)./2);
clear col_count col_sum col_mean x y val;
% Do exp function sequentially because of memory limitation
num = 2000;
num_iter = ceil(n/num);
S = sparse([]);
for i = 1:num_iter
  start_index = 1 + (i-1)*num;
  end_index = min(i*num, n);
  S1 = spfun(@exp, A(:,start_index:end_index)); % sparse exponential func
  S = [S S1];
  clear S1;
end
A = real(S);
clear S;

A = A + eps*speye(n);
A = max(A,A');
D = diag(sum(A,2));
Dd = diag(D);
clear D;
Dn=spdiags(sqrt(1./Dd),0,n,n);
An = Dn*A*Dn;
clear Dn;
An = max(An,An');



