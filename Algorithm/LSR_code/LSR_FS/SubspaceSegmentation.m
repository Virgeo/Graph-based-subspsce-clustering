function [acc,L] = SubspaceSegmentation( SegmentatiomMethod , X , gnd , para )

switch SegmentatiomMethod

    case 'LSR1'
        L = LSR1( X , para ) ;
        
    case 'LSR2'
        L = LSR2( X , para ) ;
end


for i = 1 : size(L,2)
   L(:,i) = L(:,i) / max(abs(L(:,i))) ;    
end

nCluster = length( unique( gnd ) ) ;
Z = ( abs(L) + abs(L') ) / 2 ;
idx = clu_ncut(Z,nCluster) ;
acc = compacc(idx,gnd) ;























% for i = 1 : size(L,2)
%    Z(:,i) = Z(:,i) / max(max(Z)) ;    
% end
% % Z(Z>0.01) = 1 ;
% imshow(1-Z)



% figure
% for i = 1 : size(L,2)
%    L(:,i) = L(:,i) / max(L(:,i)) ;    
% end
% imshow(1-L)


