function [] = latlrr_face_recog()
data = load('yaleball_reD.mat');
data = data.obj;

%solve latlrr
[Z,L,E] = solve_latlrr(data.Xtrain,0.4,1,1.5);

ytrain = L*data.Xtrain;
ytest = L*data.Xtest;

ks = 1:2:5;
for i=1:length(ks)
    k = ks(i);
    ypred = knnclassify(ytest',ytrain',data.traincids',k);
    acc = sum(abs(ypred'-data.testcids)<0.1)/length(ypred);
    disp(['acc(' num2str(k) 'nn)=' num2str(acc)]);
end