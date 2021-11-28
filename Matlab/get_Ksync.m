function model = get_Ksync(data)
    [NN, obs] = size(data)
    model = maxent.createModel(NN,'ksync'); %create model 
    p_k = maxent.getEmpiricalMarginals(data, model); % get marignals
    % to map matlab weights to ours: V(K) = -ln Z - ln(P(K) / N choose K )
    combs = arrayfun(@(k) nchoosek(NN,k), 0:NN); % this might be inaccurate at large NN
    null = 1e-20; % set zero probability values to 1e-20
    arg = p_k ./ combs;
    arg(arg==0) = null;
    v_k = - log(arg);
    model = maxent.setFactors(model, v_k);
    model.z = 1; %alternatively set Z to 1
end