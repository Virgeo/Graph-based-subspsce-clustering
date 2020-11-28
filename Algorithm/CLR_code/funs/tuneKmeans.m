function [finalInd, Ind, kmobj, minob] = tuneKmeans(M, Ini)

minob = 1e5;
nIni = size(Ini, 2);
kmobj = zeros(nIni);

for ii = 1 : nIni
    [Ind(:, ii), iter_num, sumd, center, obj] = kmeans(M', Ini(:, ii));
    kmobj(ii) = obj;
    if obj < minob
        minob = obj;
        finalInd = Ind(:, ii);
    end;
end;

end

