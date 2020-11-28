function E = solve_E(Q,v)
[m,n] = size(Q);
for i = 1:n
    if v < norm(Q(:,i))
        E(:,i) = (norm(Q(:,i)) - v)/norm(Q(:,i))*Q(:,i);
    else
        E(:,i) = zeros(m,1);
    end
end