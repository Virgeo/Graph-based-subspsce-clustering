function [acc] = compacc(idx1,idx0)
%inputs:
%      idx1 -- the clustering results
%      idx0 -- the groudtruth clustering results
%outputs:
%      acc -- segmentation accuracy (or classification accuracy)
uids = unique(idx1);
idx = idx1;
for i=1:length(uids)
    uid = uids(i);
    inds = abs(idx1-uid)<0.1;
    vids = idx0(inds);
    uvids = unique(vids);
    vf = 0;
    for j=1:length(uvids)
        vfj = sum(abs(vids-uvids(j))<0.1);
        if vfj>vf;
            vid = uvids(j);
            vf = vfj;
        end
    end
    idx(inds) = vid;
end
acc = sum(abs(idx-idx0)<0.1)/length(idx0);