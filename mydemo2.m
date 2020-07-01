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
    
    load([dpath, 'mData/Fmatrix/', dname, '_fea.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning');
    

    %% create clustering datasets with bag-of-words pipeline
    numclass = length(unique(Y));
    numview = length(view_meaning);
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

            discriptors = reshape(Xorg, size(Xorg,1)*size(Xorg,2), size(Xorg,3)); % (N x N_d) x d
            nvec = size(discriptors,1);
            if nvec>p.num_desc
                idx = randperm(nvec);
                tmpDisc = discriptors(idx(1:p.num_desc), :);
            else
                tmpDisc = discriptors;
            end
            dictionary = kmeansFast(tmpDisc, p.dictionary_size*10);
            p.dictionary = dictionary;
            save([dpath, 'mData/Fmatrix/', dname, '_dictionary.mat'], 'data_name', 'dictionary', '-v7.3');

            % extract llc feat
            fprintf('\n- extract llc feat.');
            llcfeat = cell(size(Xorg,1),1);
            llcknn = p.llcknn;
            parfor j=1:size(Xorg,1)
                llcfeat{j} = sparse(LLC_coding_appr(dictionary, squeeze(Xorg(j,:,:)), llcknn));
            end
            % max pooling
            fprintf('\n- max pooling.');
            poolfeat = max_pooling(llcfeat, infos{v}, p.pyramid_levels);
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