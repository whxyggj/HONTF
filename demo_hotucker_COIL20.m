function result = demo_hotucker_COIL20(nRun)
% HOTucker on complete COIL20. Pass 10 to repeat training ten times.
if nargin < 1, nRun = 1; end
validateattributes(nRun,{'numeric'},{'scalar','integer','positive'});
root_dir = setup_hontf();
rng(0,'twister');
options.lambda = 1000;
options.beta = 1;
options.maxIter = 200;
metrics_all = zeros(7,nRun);
metrics_unlabeled = zeros(7,nRun);
training_seconds = zeros(1,nRun);
objective_history = cell(1,nRun);
label_history = cell(1,nRun);
split_history = cell(1,nRun);
for run_idx = 1:nRun
    [Xl,Xu,Yl,Yu,split] = prepare_coil20(root_dir);
    k = numel(unique(Yl));
    fprintf('HOTucker: run %d/%d, %d labeled and %d unlabeled samples.\n', ...
        run_idx,nRun,numel(Yl),numel(Yu));
    tic;
    [A,B,C,G,nIter,obj_all] = hotucker(Xl,Xu,Yl,Yu,32,32,32,32,k,options);
    G3 = tenmat(tensor(G),3);
    representation = double(C*G3);
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
result.method = 'HOTucker';
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
    result.G = double(G);
result.last_nIter = nIter;
result.last_update_count = numel(obj_all);
output_dir = fullfile(root_dir,'results');
if ~exist(output_dir,'dir'), mkdir(output_dir); end
save(fullfile(output_dir,'HOTucker_COIL20.mat'),'result');
fprintf('HOTucker all samples: ACC %.4f, NMI %.4f, ARI %.4f\n', ...
    result.mean_all(1),result.mean_all(2),result.mean_all(7));
fprintf('HOTucker unlabeled:   ACC %.4f, NMI %.4f, ARI %.4f\n', ...
    result.mean_unlabeled(1),result.mean_unlabeled(2),result.mean_unlabeled(7));
fprintf('Last run: %d recorded updates; %.2f s training. Saved in %s\n', ...
    result.last_update_count,training_seconds(end),output_dir);
end
