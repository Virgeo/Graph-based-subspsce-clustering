clc;
close all;

folder_now = pwd;  addpath([folder_now, '\funs']);

newdata = 1;
datatype = 2; % 1: two-moon data, 2: three-ring data, 3: 196-cluster data

if newdata == 1
    clearvars -except datatype;
    
    if datatype == 1
        num0 = 100; X = twomoon_gen(num0); c = 2; y = [ones(num0,1);2*ones(num0,1)];
    elseif datatype == 2
        num0 = 500; [X, y] = three_ring_gen(num0,0.05); c = 3;
    else
        c=196; [X, y] = spheres_gen(c, c*10, 0.03);
    end;
end;


[la, A, evs] = CAN(X', c);
figure('name','Learned graph by CAN'); 
imshow(A,[]); colormap jet; colorbar;

cm = colormap(jet(c));
figure('name','Clustering results by CAN');
plot(X(:,1),X(:,2),'.k'); hold on;

rl = randperm(c);
for i=1:c
    plot(X(la==rl(i),1),X(la==rl(i),2),'.', 'color', cm(i,:)); hold on;
end;
result_can = ClusteringMeasure(y, la);
[ind,sumd,center, ob_can] = kmeans_ldj(X,la);
result_cankm0 = ClusteringMeasure(y, ind);
result_cankm = [result_cankm0(1), ob_can]

% km clustering
if datatype==3
    for i=1:100
        [ind0(:,i),sumd,center0(:,:,i),ob(i)] = kmeans_ldj(X,c);
    end;
    ob1=sort(ob);
    [obkm_best, idx_best] = min(ob);
    y1 = ind0(:,idx_best);
    result_km0 = ClusteringMeasure(y, y1);
    result_km = [result_km0(1), obkm_best]

    cm = colormap(jet(c));
    figure('name','Clustering results by Kmeans');
    rl = randperm(c);
    for i=1:c
        plot(X(y1==rl(i),1),X(y1==rl(i),2),'.', 'color', cm(i,:)); hold on;
    end;

end;



