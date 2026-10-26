function [Xl,Xu,Yl,Yu,split] = prepare_coil20(root_dir)
% Original demo sampling and label selection, using all 20 COIL20 classes.
d = load(fullfile(root_dir,'data','COIL20.mat'));
fea = NormalizeFea(d.fea);
gnd = d.gnd(:);
classes = unique(gnd);
k = numel(classes);
nEach = 72;
assert(isequal(size(fea),[1440,1024]) && k == 20, 'Unexpected COIL20 data.');
shuffleClasses = randperm(k);
selected = zeros(nEach*k,1);
labels = zeros(nEach*k,1);
for j = 1:k
    idx = find(gnd == classes(shuffleClasses(j)));
    rows = (j-1)*nEach+(1:nEach);
    selected(rows) = idx(1:nEach);
    labels(rows) = j;
end
samples = fea(selected,:);
semiSplit = false(size(labels));
for j = 1:k
    idx = find(labels == j);
    shuffleIndexes = randperm(numel(idx));
    semiSplit(idx(shuffleIndexes(1:floor(0.3*numel(idx))))) = true;
end
Xl = samples(semiSplit,:)';
Xu = samples(~semiSplit,:)';
Yl = labels(semiSplit);
Yu = labels(~semiSplit);
split.sample_indices = [selected(semiSplit);selected(~semiSplit)];
split.labels = [Yl;Yu];
split.is_labeled = [true(numel(Yl),1);false(numel(Yu),1)];
split.original_class_order = classes(shuffleClasses);
split.requested_labeled_fraction = 0.3;
split.actual_labeled_fraction = mean(semiSplit);
end
