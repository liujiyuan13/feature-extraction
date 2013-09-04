function [feat] = batch_feature(filelist, imgset, feature, c)
if(~exist('c', 'var'))
  c = conf();
end

p = c.feature_config.(feature);
if(isfield(p, 'dictionary_size'))
	feature_file = sprintf(p.([imgset '_file']), c.cache, p.dictionary_size);
else
    feature_file = sprintf(p.([imgset '_file']), c.cache);
end

if(exist(feature_file, 'file'))
  load(feature_file);
  return;
end

num_batches = ceil(length(filelist)/c.batch_size);
batch_idx = arrayfun(@(x) (x-1)*c.batch_size+1:min(x*c.batch_size, length(filelist)), 1:num_batches, 'UniformOutput', false);
batch_order = randperm(num_batches);
batch_files = cell(num_batches, 1);

for b=1:num_batches
    this_batch = batch_idx{batch_order(b)};
    batch_file = [c.cache imgset '_' feature '_' num2str(p.dictionary_size) '/' num2str(batch_order(b)) '.mat'];
    batch_files{batch_order(b)} = batch_file;
    fprintf('Processing filelist (%s, %s): batch %d of %d\n', imgset, feature, b, num_batches);
    if(~exist(batch_file, 'file'))
        parsaveFeat(batch_file, [], []);
        poolfeat = filelist_feature('', filelist(this_batch), feature, c);
        parsaveFeat(batch_file, poolfeat, filelist(this_batch));
    end
end

if(nargout>0)
    feat = cell(num_batches, 1);
    for i=1:num_batches
        tmp = load(batch_files{i});
        feat{i} = tmp.poolfeat;
    end
    feat = cell2mat(feat);
else
    feat = {};
end

save(feature_file, 'feat', 'batch_files', '-v7.3');