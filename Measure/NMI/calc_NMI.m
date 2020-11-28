%������������֮��Ļ���Ϣ
%u1��������������1
%u2��������������2
%wind_size��������ά��

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
    [occur,bin(:,i)] = histc(x(:,i),histcEdges,1); %ͨ��ֱ��ͼ��ʽ���㵥��������ֱ��ͼ�ֲ�
    pmf(:,i) = occur(1:n)./xrow;
end
%����u1��u2�����ϸ����ܶ�
jointOccur = accumarray(bin,1,[n,n]);  %��xi��yi����������ͬʱ����n*n�ȷַ����е�������Ϊ���ϸ����ܶ�
jointPmf = jointOccur./xrow;
Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);
Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);
Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);
MI = Hx+Hy-Hxy;
NMI = MI/max(Hx,Hy);
