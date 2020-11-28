%计算两列向量之间的互信息
%u1：输入计算的向量1
%u2：输入计算的向量2
%wind_size：向量的维度

function NMI = calc_NMI(u1, u2, wind_size)
x = [u1, u2];
n = wind_size;
[xrow, xcol] = size(x);
bin = zeros(xrow,xcol);
pmf = zeros(n, 2);
for i = 1:2
    minx = min(x(:,i));
    maxx = max(x(:,i));
    binwidth = (maxx - minx) / n;
    edges = minx + binwidth*(0:n);
    histcEdges = [-Inf edges(2:end-1) Inf];
    [occur,bin(:,i)] = histc(x(:,i),histcEdges,1); %通过直方图方式计算单个向量的直方图分布
    pmf(:,i) = occur(1:n)./xrow;
end
%计算u1和u2的联合概率密度
jointOccur = accumarray(bin,1,[n,n]);  %（xi，yi）两个数据同时落入n*n等分方格中的数量即为联合概率密度
jointPmf = jointOccur./xrow;
Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);
Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);
Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);
MI = Hx+Hy-Hxy;
NMI = MI/max(Hx,Hy);
