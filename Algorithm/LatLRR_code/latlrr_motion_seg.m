function [] = latlrr_motion_seg()
tic;
datadir = '../datas/hop155';
seqs = dir(datadir);
seq3 = seqs(3:end);
%% load the data
data = struct('X',{},'name',{},'ids',{});
for i=1:length(seq3)
    fname = seq3(i).name;
    fdir = [datadir '/' fname];
    if isdir(fdir)
        datai = load([fdir '/' fname '_truth.mat']);
        id = length(data)+1;
        data(id).ids = datai.s;
        data(id).name = lower(fname);
        X = reshape(permute(datai.x(1:2,:,:),[1 3 2]),2*datai.frames,datai.points);
        data(id).X = [X;ones(1,size(X,2))];
    end
end
clear seq3;
%% preprocessing
rand('state', 1212498032853324);
randn('state', 121243456980328533249450);
for i=1:length(data)
    X = data(i).X;
    [d n] = size(X);
    noise = randn(d,n);
    noise = 0.03*noise.*abs(X);
    inds = rand(d,n)<10/100;
    X(inds) = X(inds)+noise(inds);
    data(i).X = X;
end
%% segmentation 
errs = zeros(length(data),1);
for i=1:length(data)
    X = data(i).X;
    gnd = data(i).ids; K = max(gnd);
    if abs(K-2)>0.1 && abs(K-3)>0.1
        id = i; % the discarded sequqnce
    end
    % perfrom latlrr
    Z = solve_latlrr(X,1.4);
    
    %
    disp('post processing ... ...');
    Z = rpcapsd(Z,0.5);
    
    % refining Z
    [U,S,V] = svd(Z);
    S = diag(S);
    r = min(4*K+1,sum(S>1e-3*S(1)));
    S = S(1:r);
    U = U(:,1:r)*diag(sqrt(S));
    U = normr(U);
    Z = U*U';Z=abs(Z);L = Z.^4.5;
        
    % spectral clustering
    L = (L + L')/2;
    D = diag(1./sqrt(sum(L,2)));
    L = D*L*D;
    [U,S,V] = svd(L);

    V = U(:,1:K);
    V = D*V;
    idx = kmeans(V,K,'emptyaction','singleton','replicates',20,'display','off');
      
    % display
    err =  missclassGroups(idx,gnd,K)/length(idx);
    disp(['seq ' num2str(i) ',err=' num2str(err)]);
    errs(i) = err;
end
disp('results of all 156 sequences:');
disp(['max = ' num2str(max(errs)) ',min=' num2str(min(errs)) ...
    ',median=' num2str(median(errs)) ',mean=' num2str(mean(errs)) ',std=' num2str(std(errs))] );

errs = errs([1:id-1,id+1:end]);
disp('results of all 155 sequences:');
disp(['max = ' num2str(max(errs)) ',min=' num2str(min(errs)) ...
    ',median=' num2str(median(errs)) ',mean=' num2str(mean(errs)) ',std=' num2str(std(errs))] );

toc;
