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

for dn = 1:1
    dname = dnames{dn};
    
    fprintf('# handling with %s', dname);
    
    load([dpath, 'mData/Fmatrix/', dname, '_fea.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning');
    
    %% create AD datasets with bag-of-words pipline
    % obtain the train-test split
    numclass = length(unique(Y));
    numview = length(view_meaning);
    for i=1:numclass
        norm_label = int32(i);
        ind = find(Y==i);
        rind = ind(randperm(length(ind)));
        numtra = floor(length(rind)*0.8);
        rindtra = rind(1:numtra);
        rindtes = rind(numtra+1:end);
        indad = [rindtes; find(Y~=i)];

        Xtra = cell(numview, 1);
        Xtes = cell(numview, 1);
        for v=1:numview
            feature = view_meaning{v};
            fprintf('\n# feature: %s', feature);
            if strcmp(view_meaning{v}, 'ssim')
                tmpinfo = infos{v};
                parfor s=1:length(Y)
                    tmpinfo{s}.x = tmpinfo{s}.x(:);
                    tmpinfo{s}.y = tmpinfo{s}.y(:);
                end
                infos{v} = tmpinfo;
            end
            if ~strcmp(view_meaning{v},'gist') && ~strcmp(view_meaning{v}, 'lbp')
                % build dictionary
                fprintf('\n- build dictionary.');
                c.feature_config.(feature) = feval(['config_',feature], c);
                p = c.feature_config.(feature);
                Xtmp = reshape(X{v}, Fshape{v}(2), Fshape{v}(1), size(X{v},2));
                Xtmp = permute(Xtmp, [3,2,1]); % N x N_d x d
                X_dict = Xtmp(rindtra,:,:);

                discriptors = reshape(X_dict, size(X_dict,1)*size(X_dict,2), size(X_dict,3)); % (N x N_d) x d
                nvec = size(discriptors,1);
                if nvec>p.num_desc
                    idx = randperm(nvec);
                    tmpDisc = discriptors(idx(1:p.num_desc), :);
                else
                    tmpDisc = discriptors;
                end
                dictionary = kmeansFast(tmpDisc, p.dictionary_size);
                p.dictionary = dictionary;
                save([dpath, 'mData/ADmatrix/', dname, '/', dname, '_', feature, '_dictionary_', int2str(i), '.mat'], 'data_name', 'norm_label', 'dictionary', '-v7.3');

                % extract llc feat
                fprintf('\n- extract llc feat.');
                llcfeat = cell(size(Xtmp,1),1);
                llcknn = p.llcknn;
                parfor j=1:size(Xtmp,1)
                    llcfeat{j} = sparse(LLC_coding_appr(dictionary, squeeze(Xtmp(j,:,:)), llcknn));
                end
                % max pooling
                fprintf('\n- max pooling.');
                % feat of norm class
                infos_tra = infos{v}(rindtra)';
                llcfeat_tra = llcfeat(rindtra);
                poolfeat_tra = max_pooling(llcfeat_tra, infos_tra, c.pool_region, p.pyramid_levels);
                poolfeat_tra = cast(poolfeat_tra, c.precision);
                % feat of ad class
                infos_ad = infos{v}(indad)';
                llcfeat_ad = llcfeat(indad);
                poolfeat_ad = max_pooling(llcfeat_ad, infos_ad, c.pool_region, p.pyramid_levels);
                poolfeat_ad = cast(poolfeat_ad, c.precision);

                Xtra{v} = double(poolfeat_tra');
                Xtes{v} = double(poolfeat_ad');
            else
                Xtra{v} = double(X{v}(:,rindtra));
                Xtes{v} = double(X{v}(:,indad));
            end
        end
        Ytra = int32(Y(rindtra));
        Ytes = int32(Y(indad));

        save([dpath, 'mData/ADmatrix/', dname, '/', dname, 'AD_fea_', int2str(i), '.mat'], 'data_name', 'Xtra', 'Xtes', 'Ytra', 'Ytes', 'rindtra', 'indad', 'norm_label', 'class_meaning', 'view_meaning', '-v7.3');

    end

    
end