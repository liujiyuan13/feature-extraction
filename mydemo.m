clear
clc
warning('off');

% addpath(genpath(pwd));
c = conf();
openPool(c.cores);

% dpath = 'D:\Work\datasets\mData\OrigData\';
dpath = '/home/ftp2/jiyuan/datasets/';
dnames = {'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'STL10'};
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
    Fshape = cell(numfeat, 1);
    for f = 1:numfeat
        feature = view_meaning{f};
        tmpF = zeros(0);
        tmpinfo = cell(0);
        parfor i = 1:numsmp
            img = squeeze(X(i,:,:,:));
            [tmp, tmpinfo{i}.x, tmpinfo{i}.y, tmpinfo{i}.wid, tmpinfo{i}.hgt] = extract_feature(feature, img);
%             Fshape{1} = [size(tmp,1), size(tmp,2)]; % N_d x d
            tmpF(:,i) = reshape(tmp', size(tmp,1)*size(tmp,2), 1);
        end
        Xtmp{f} = double(tmpF);
        infos{f} = tmpinfo;
    end
    class_meaning = class_meaning';
    data_name = dname;

    fprintf('\n\n# save fea mat of %s', dname);
    
%     save([dpath, dname, '\', dname, '_img.mat'], 'data_name', 'X', 'Y', 'class_meaning', 'view_meaning', '-v7.3')
    save([dpath, 'mData/Fmatrix/', dname, '_img.mat'], 'data_name', 'X', 'Y', 'infos', 'class_meaning', 'view_meaning', '-v7.3');
    
    % manually add Fshape
%     save([dpath, 'mData/Fmatrix/', dname, '_img.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning', '-v7.3');
            
    
end