% Motion segmentation on the Hopkins 155 database by using LSR in

% Can-Yi Lu, Hai Min, Zhong-Qiu Zhao, Lin Zhu, De-Shuang Huang and Shuicheng Yan. 
% Robust and Efficient Subspace Segmentation via Least Squares Regression,
% European Conference on Computer Vision (ECCV), 2012.

%--------------------------------------------------------------------------
% Copyright @ Can-Yi Lu, 2012
%--------------------------------------------------------------------------

clear ;
% close all;
cd '/Users/CWG/Documents/m.text/Hopkins155/'; 
currentpath =  '/Users/CWG/Documents/m.text/SSC_motion_face/';
AddedPath = genpath( currentpath ) ;
addpath( AddedPath ) ;
fprintf('\n\n**************************************   %s   *************************************\n' , datestr(now) );
fprintf( [ mfilename(currentpath) ' Begins.\n' ] ) ;
fprintf( [ mfilename(currentpath) ' is going, please wait...\n' ] ) ;

%% reduced dimension
ProjRank = 12 ;

seqs = dir;
% Get rid of the two directories: "." and ".."
seq3 = seqs(3:end);
% Save the data loaded in struct "data"
data = struct('ProjX', {}, 'name',{}, 'ids',{});


for i=1:length(seq3)
    fname = seq3(i).name;
    fdir = [cd '/' fname];
    if isdir(fdir)
        datai = load([fdir '/' fname '_truth.mat']);
        id = length(data)+1;
        % the true group numbers
        data(id).ids = datai.s;
        % file name
        data(id).name = lower(fname);
        % X is the motion sequence
        X = reshape(permute(datai.x(1:2,:,:),[1 3 2]), 2*datai.frames, datai.points);
        
        % PCA projection
        [ eigvector , eigvalue ] = PCA( X ) ;    
        ProjX = eigvector(:,1:ProjRank)' * X ;     
        data(id).ProjX = [ProjX ; ones(1,size(ProjX,2)) ] ;
    end
end
clear seq3;


%% Subspace segmentation methods
SegmentationMethod = 'LSR1' ;     % LSR1 by (16) in our paper
% SegmentationMethod = 'LSR2' ;     % LSR2 by (18) in our paper


%% Parameter
switch SegmentationMethod    
    case 'LSR1'
        para = 4.8*1e-3 ;
    case 'LSR2'
        para = 4.6*1e-3 ;
end


%% 
errs = zeros(length(data),1);
for i = 1 : 156 % 156 sequences
    ProjX = data(i).ProjX ;
    gnd = data(i).ids' ; 
    K = length( unique( gnd ) ) ;
    errs(i) = SubspaceSegmentation( SegmentationMethod , ProjX , gnd , para ) ;
    errs(i) = 100 - 100 * errs(i) ;
    fprintf('seq %d\t %f\n', i , errs(i) ) ;
end
fprintf('\n') ;


%% 
err_mean = mean(errs) ;
err_max = max(errs) ;
err_min = min(errs) ;
err_median = median(errs) ;
err_std = std(errs) ;

para
disp(['max = ' num2str(max(errs)) ',min=' num2str(min(errs)) ...
    ',median=' num2str(median(errs)) ',mean=' num2str(mean(errs)) ',std=' num2str(std(errs))] );





