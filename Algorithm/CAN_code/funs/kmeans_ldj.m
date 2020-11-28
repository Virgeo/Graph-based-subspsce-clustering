function [Ind, sumd, center, obj] = kmeans_ldj(M,StartIndMeanK)
% each row is a data point

[nSample, nFeature] = size(M);
if isscalar(StartIndMeanK)
%     t = randperm(nSample);
%     StartIndMeanK = t(1:StartIndMeanK);
    StartIndMeanK = randsrc(nSample,1,1:StartIndMeanK);
elseif nSample ~= length(StartIndMeanK)
    error('each row should be a data point');
end
if isvector(StartIndMeanK)
    K = length(StartIndMeanK);
    if K == nSample
        K = max(StartIndMeanK);
        means = zeros(K,nFeature);
        for ii=1:K
            means(ii,:) = mean(M(find(StartIndMeanK==ii),:),1);
        end
    else
        means = zeros(K,nFeature);
        for ii=1:K
            means(ii,:) = M(StartIndMeanK(ii),:);
        end
    end
else
    K = size(StartIndMeanK,1);
    means = StartIndMeanK;
end
center = means;
M2 = sum(M.*M, 2)';
twoMp = 2*M';
M2b = repmat(M2,[K,1]);
Center2 = sum(center.*center,2);Center2a = repmat(Center2,[1,nSample]);[xx, Ind] = min(abs(M2b + Center2a - center*twoMp));
Ind2 = Ind;
it = 1;
%while true
while it < 2000
    for j = 1:K
        dex = find(Ind == j);
        l = length(dex);
        if l > 1;                 center(j,:) = mean(M(dex,:));
        elseif l == 1;            center(j,:) = M(dex,:);
        else                      t = randperm(nSample);center(j,:) = M(t(1),:);
        end;
    end;
    Center2 = sum(center.*center,2);Center2a = repmat(Center2,[1,nSample]);[dist, Ind] = min(abs(M2b + Center2a - center*twoMp));
    if Ind2==Ind;       
        break;    
    end
    Ind2 = Ind;
    it = it+1;
end
sumd = zeros(K,1);
for ii=1:K
    idx = find(Ind==ii);
    dist2 = dist(idx);
    sumd(ii) = sum(dist2);
end


Ind = Ind';
obj = sum(sumd);

end



