% addpath(genpath(pwd));
c = conf();
openPool(c.cores);

% dpath = 'D:\Work\datasets\mData\OrigData\';
dpath = '/home/ftp2/jiyuan/datasets/';
dnames = {'MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'};
len_dn = length(dnames);

view_meaning = {'color', 'gist', 'hog2x2', 'hog3x3', 'lbp', 'sift', 'ssim'}';
numfeat = length(view_meaning);

for dn = 2:4
    dname = dnames{dn};
    
    fprintf('# handling with %s', dname);
    
    load([dpath, 'mData/Fmatrix/', dname, '_fea.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning');
    
    % no need ssim for it's too long
    view_meaning = view_meaning(1:6);
    
    %% create clustering datasets with bag-of-words pipeline
    numview = length(view_meaning);
    numsmp = size(X{1},2);
    Xtmp = cell(numview, 1);
    for v=1:numview
        feature = view_meaning{v};
        fprintf('\n# feature: %s', feature);
        if ~strcmp(feature,'gist') && ~strcmp(feature, 'lbp')
            % build dictionary
            fprintf('\n- build dictionary.');
            c.feature_config.(feature) = feval(['config_',feature], c);
            p = c.feature_config.(feature);
            Xorg = reshape(X{v}, Fshape{v}(2), Fshape{v}(1), size(X{v},2));
            Xorg = permute(Xorg, [3,2,1]); % N x N_d x d
            
            inddict = randperm(numsmp);
            inddict = inddict(1:floor(numsmp/10));
            Xdict = Xorg(inddict, :, :);

            discriptors = reshape(Xdict, size(Xdict,1)*size(Xdict,2), size(Xdict,3)); % (N x N_d) x d
            nvec = size(discriptors,1);
            if nvec>p.num_desc
                idx = randperm(nvec);
                tmpDisc = discriptors(idx(1:p.num_desc), :);
            else
                tmpDisc = discriptors;
            end
            dictionary = kmeansFast(tmpDisc, p.dictionary_size);
            p.dictionary = dictionary;
            save([dpath, 'mData/Fmatrix/', dname, '_', feature, '_dictionary.mat'], 'data_name', 'inddict', 'dictionary', '-v7.3');

            % extract llc feat
            fprintf('\n- extract llc feat.');
            llcfeat = cell(size(Xorg,1),1);
            llcknn = p.llcknn;
            parfor j=1:size(Xorg,1)
                llcfeat{j} = sparse(LLC_coding_appr(dictionary, squeeze(Xorg(j,:,:)), llcknn));
            end
            % max pooling
            fprintf('\n- max pooling.');
            poolfeat = max_pooling(llcfeat, infos{v}', c.pool_region, p.pyramid_levels);
            poolfeat = cast(poolfeat, c.precision);

            Xtmp{v} = double(poolfeat');
        else
            Xtmp{v} = double(X{v});
        end
    end
    Y = int32(Y);
    X = Xtmp;

    save([dpath, 'mData/Fmatrix/', dname, '_llc_fea.mat'], 'data_name', 'X', 'Y', 'class_meaning', 'view_meaning', '-v7.3');
    
end