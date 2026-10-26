function result = demo_hocp_COIL20(nRun)
% HOCP on complete COIL20. Pass 10 to repeat training ten times.
if nargin < 1, nRun = 1; end
validateattributes(nRun,{'numeric'},{'scalar','integer','positive'});
root_dir = setup_hontf();
rng(0,'twister');
options.lambda = 1000;
options.beta = 100;
options.maxIter = 500;
metrics_all = zeros(7,nRun);
metrics_unlabeled = zeros(7,nRun);
training_seconds = zeros(1,nRun);
objective_history = cell(1,nRun);
label_history = cell(1,nRun);
split_history = cell(1,nRun);
for run_idx = 1:nRun
    [Xl,Xu,Yl,Yu,split] = prepare_coil20(root_dir);
    k = numel(unique(Yl));
    fprintf('HOCP: run %d/%d, %d labeled and %d unlabeled samples.\n', ...
        run_idx,nRun,numel(Yl),numel(Yu));
    tic;
    [A,B,C,nIter,obj_all] = hocp(Xl,Xu,Yl,Yu,k,32,32,options);
    representation = C;
    training_seconds(run_idx) = toc;
    assert(isequal(size(C),[numel(Yl)+numel(Yu),k]), 'Invalid representation size.');
    assert(isreal(representation) && all(isfinite(representation(:))) ...
        && all(representation(:)>=0), 'Invalid clustering representation.');
    assert(~isempty(obj_all) && all(isfinite(obj_all)), 'Invalid iteration history.');
    predicted = litekmeans(representation,k,'Replicates',20);
    metrics_all(:,run_idx) = EvaluationMetrics([Yl;Yu],predicted)';
    metrics_unlabeled(:,run_idx) = EvaluationMetrics(Yu,predicted(numel(Yl)+1:end))';
    objective_history{run_idx} = obj_all;
    label_history{run_idx} = predicted;
    split_history{run_idx} = split;
end
result.method = 'HOCP';
result.metric_names = {'ACC','NMI','purity','Fscore','precision','recall','ARI'};
result.metrics_all = metrics_all;
result.metrics_unlabeled = metrics_unlabeled;
result.mean_all = mean(metrics_all,2);
result.std_all = std(metrics_all,0,2);
result.mean_unlabeled = mean(metrics_unlabeled,2);
result.std_unlabeled = std(metrics_unlabeled,0,2);
result.objective_history = objective_history;
result.predicted_labels = label_history;
result.splits = split_history;
result.training_seconds = training_seconds;
result.options = options;
result.seed = 0;
result.nRun = nRun;
result.A = A; result.B = B; result.C = C;
result.last_nIter = nIter;
result.last_update_count = numel(obj_all);
output_dir = fullfile(root_dir,'results');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'HOCP_COIL20.mat'),'result');
fprintf('HOCP all samples: ACC %.4f, NMI %.4f, ARI %.4f\n', ...
    result.mean_all(1),result.mean_all(2),result.mean_all(7));
fprintf('HOCP unlabeled:   ACC %.4f, NMI %.4f, ARI %.4f\n', ...
    result.mean_unlabeled(1),result.mean_unlabeled(2),result.mean_unlabeled(7));
fprintf('Last run: %d recorded updates; %.2f s training. Saved in %s\n', ...
    result.last_update_count,training_seconds(end),output_dir);
end
