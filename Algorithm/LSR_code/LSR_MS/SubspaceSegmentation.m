function acc = SubspaceSegmentation( SegmentatiomMethod , X , gnd , para )

switch SegmentatiomMethod

    case 'LSR1'
        L = LSR1( X , para ) ;
        
    case 'LSR2'
        L = LSR2( X , para ) ;
end

nCluster = length( unique( gnd ) ) ;
Z = ( abs(L) + abs(L') ) / 2 ;
idx = clu_ncut(Z,nCluster) ;
acc = compacc(idx,gnd) ;
