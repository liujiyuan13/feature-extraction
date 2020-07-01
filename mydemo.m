% addpath(genpath(pwd));
c = conf();
openPool(c.cores);

% dpath = 'D:\Work\datasets\mData\OrigData\';
dpath = '/home/ftp2/jiyuan/datasets/';
dnames = {'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'};
len_dn = length(dnames);

view_meaning = {'color', 'gist', 'hog2x2', 'hog3x3', 'lbp', 'sift', 'ssim'}';
numfeat = length(view_meaning);

for dn = 1:1
    dname = dnames{dn};
    
    fprintf('# handling with %s', dname);
    
%     load([dpath, dname, '\', dname, '_img.mat'], 'X', 'Y', 'class_meaning');
    load([dpath, 'mData/Imatrix/', dname, '_img.mat'], 'X', 'Y', 'class_meaning');
    
    numsmp = size(X, 1);
    parfor i = 1:numsmp
        warning('off');
        img = squeeze(X(i,:,:,:));
        tmp = extract_feature('color', img);
        color(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('gist', img);
        gist(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('hog2x2', img);
        hog2x2(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('hog3x3', img);
        hog3x3(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('lbp', img);
        lbp(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('sift', img);
        sift(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
        tmp = extract_feature('ssim', img);
        ssim(:,i) = reshape(tmp, size(tmp,1)*size(tmp,2), 1);
    end
    
    X = {color, gist, hog2x2, hog3x3, lbp, sift, ssim}';
%     X = {color, gist, lbp, sift, ssim}';
    class_meaning = class_meaning';
    data_name = dname;

    fprintf('# save fea mat of %s', dname);
    
%     save([dpath, dname, '\', dname, '_img.mat'], 'data_name', 'X', 'Y', 'class_meaning', 'view_meaning', '-v7.3')
    save([dpath, 'mData/Fmatrix/', dname, '_img.mat'], 'data_name', 'X', 'Y', 'class_meaning', 'view_meaning', '-v7.3')
    
end