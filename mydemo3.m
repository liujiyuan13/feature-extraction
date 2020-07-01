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
    
    save([dpath, 'mData/Fmatrix/', dname, '_img.mat'], 'data_name', 'X', 'Y', 'infos', 'Fshape', 'class_meaning', 'view_meaning', '-v7.3')
    
    %% create AD datasets with bag-of-words pipline
    % obtain the train-test split
    data_name;
    X;
    numclass = length(unique(Y));
    class_meaning;
    view_meaning;
    numview = length(view_meaning);
    for i=1:numclass
        ind = find(Y==i);
        rind = ind(randperm(length(ind)));
        numtra = floor(length(rind));
        rindtra = rind(1:numtra);
        rindtes = rind(numtra+1,end);
        indad = [rindtes, find(Y~=i)];

        Xtra = cell(numview, 1);
        Xtes = cell(numview, 1);
        for v=1:numview
            if ~strcmp(view_meaning{v},'gist') && ~strcmp(view_meaning{v}, 'lbp')
                % build dictionary
                feature = view_meaning{v};
                c.feature_config.(feature) = feval(['config_',feature], c);
                p = c.feature_config.(feature);
                Xtmp = reshape(X{v}, Fshape{v}(2), Fshape{v}(1), size(X{v},2));
                Xtmp = permute(Xtmp, [3,2,1]); % N x N_d x d
                X_dict = Xtmp(rindtra,:,:);

                discriptors = reshape(X_dict, size(X_dict,1)*size(X_dict,2), size(X_dict,3)); % (N x N_d) x d
                nvec = size(discriptors,1);
                if nvec>p.num_desc
                    idx = randperm(nvec);
                    tmpDisc = discriptors(idx(1:p.num_desc,:), :);
                else
                    tmpDisc = discriptors;
                end
                dictionary = kmeansFast(tmpDisc, p.dictionary_size);
                p.dictionary = dictionary;
                save([], 'dictionary');

                % extract llc feat
                llcfeat = cell(size(Xtmp,1),1);
                for j=1:size(Xtmp,1)
                    llcfeat{j} = sparse(LLC_coding_appr(dictionary, squeeze(Xtmp(j,:,:)), p.llcknn));
                end
                % feat of norm class
                infos_tra = infos{v}{rindtra};
                llcfeat_tra = llcfeat{rindtra};
                poolfeat_tra = max_pooling(llcfeat_tra, infos_tra, p.pyramid_levels);
                poolfeat_tra = cast(poolfeat_tra, c.precision);
                % feat of ad class
                infos_ad = infos{v}{indad};
                llcfeat_ad = llcfeat{indad};
                poolfeat_ad = max_pooling(llcfeat_ad, infos_ad, p.pyramid_levels);
                poolfeat_ad = cast(poolfeat_ad, c.precision);

                Xtratmp = poolfeat_tra';
                Xtestmp = poolfeat_ad';
            else
                Xtratmp = X{v}(:,rindtra);
                Xtesemp = X{v}(:,indad);
            end
            Xtra{v} = double(Xtratmp);
            Xtes{v} = double(Xtestmp);
        end
        Ytra = int32(Y(rindtra));
        Ytes = int32(Y(indad));

        norm_label = int32(i);
        save(['', ], 'data_name', 'Xtra', 'Xtes', 'Ytra', 'Ytes', 'norm_label', 'class_meaning', 'view_meaning', '-v7.3');

    end

    
end