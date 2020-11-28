%--------------------------------------------------------------------------
% [miss,index] = missclass(Segmentation,RefSegmentation,ngroups)
% Computes the number of missclassified points in the vector Segmentation. 
% Segmentation: 1 by sum(npoints) or sum(ngroups) by 1 vector containing 
% the label for each group, ranging from 1 to n
% npoints: 1 by ngroups or ngroups by 1 vector containing the number of 
% points in each group.
% ngroups: number of groups
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function [miss,index] = missclassGroups(Segmentation,RefSegmentation,ngroups)
if(size(RefSegmentation,2)==1)
    RefSegmentation=RefSegmentation';
end
if(size(Segmentation,2)==1)
    Segmentation=Segmentation';
end
Permutations = perms(1:ngroups);
miss = zeros(size(Permutations,1),size(Segmentation,1));
for k=1:size(Segmentation,1)
    for j=1:size(Permutations,1)
        miss(j,k) = sum(abs(Segmentation(k,:)-Permutations(j,RefSegmentation))>0.1);
    end
end

[miss,temp] = min(miss,[],1);
index = Permutations(temp,:);
