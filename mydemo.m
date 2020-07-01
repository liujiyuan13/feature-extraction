clear
clc
warning('off');

% addpath(genpath(pwd));
c = conf();
openPool(c.cores);

% dpath = 'D:\Work\datasets\mData\OrigData\';
dpath = '/home/ftp2/jiyuan/datasets/';
dnames = {'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10'};
Fshapes = {{[77,50], [1,512], [25,124], [25,279], [1,1239], [26,128], [784,30]}, ...
          {[77,50], [1,512], [25,124], [25,279], [1,1239], [26,128], [784,30]}, ...
          {[110,50], [1,512], [36,124], [36,279], [1,1239], [38,128], [1024,30]}, ...
          {[110,50], [1,512], [36,124], [36,279], [1,1239], [38,128], [1024,30]}, ...
          {[1454,50], [1,512], [256,124], [256,279], [1,1239], [590,128], [9216,30]}};
len_dn = length(dnames);

view_meaning = {'color', 'gist', 'hog2x2', 'hog3x3', 'lbp', 'sift', 'ssim'}';
numfeat = length(view_meaning);

for dn = 1:len_dn
    dname = dnames{dn};
    
    fprintf('\n\n# handling with %s', dname);
    
%     load([dpath, dname, '\', dname, '_img.mat'], 'X', 'Y', 'class_meaning');
    load([dpath, 'mData/Imatrix/', dname, '_img.mat'], 'X', 'Y', 'class_meaning');

    numsmp = size(X, 1);
    
    Xtmp = cell(numfeat, 1);
    infos = cell(numfeat, 1);
    for f = 1:numfeat
        feature = view_meaning{f};
        fprintf('\n# feature: %s', feature);
        tmpF = zeros(0);
        tmpinfo = cell(0);
        parfor i = 1:numsmp
            img = squeeze(X(i,:,:,:));
            [tmp, tmpinfo{i}.x, tmpinfo{i}.y, tmpinfo{i}.wid, tmpinfo{i}.hgt] = extract_feature(feature, img);
%             Fshape{f} = [size(tmp,1), size(tmp,2)]; % N_d x d
            tmpF(:,i) = reshape(tmp', size(tmp,1)*size(tmp,2), 1);
        end
        Xtmp{f} = double(tmpF);
        infos{f} = tmpinfo;
    end
    data_name = dname;
    Fshape = Fshapes{dn}';
    X = Xtmp;

    fprintf('\n\n# save fea mat of %s', dname);
    
    save([dpath, 'mData/Fmatrix/', dname, '_fea.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning', '-v7.3');
            
    
end