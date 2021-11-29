%% load a subset of 10 neurons from the data
clear
load data2014

% explore data
size(train_rand)

% we are going to use the data where all observations over all repeats were shuffled
NN = 10;
train = train_rand(1:NN,:); % select activity of the first 10 neurons
test = test_rand(1:NN,:);

size(train)


% ============= get expectations

%% ============= independent model
m_indep = maxent.createModel(NN, 'indep'); %create model
m_indep = maxent.trainModel(m_indep, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_indep)
data_marginals = maxent.getEmpiricalMarginals(train, m_indep)

% get model weights
weights = maxent.getFactors(m_indep);
writematrix(weights, 'ModelWeights/indep.csv')

dist = maxent.getExplicitDistribution(m_indep);
writematrix(dist, 'ModelWeights/indep_dist_2.csv')



%% ============= p(K)
m_ksync = maxent.createModel(NN,'ksync'); %create model
m_ksync = maxent.trainModel(m_ksync, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_ksync)
data_marginals = maxent.getEmpiricalMarginals(train, m_ksync)

% get model weights
weights = maxent.getFactors(m_ksync);
writematrix(weights, 'ModelWeights/ksync.csv')

% get explicit distribution
dist = maxent.getExplicitDistribution(m_ksync);
writematrix(dist, 'ModelWeights/ksync_dist.csv')


%% ============= pairwise model
m_pwise = maxent.createModel(NN,'pairwise');
m_pwise = maxent.trainModel(m_pwise, train); %train model

% compare marginals
m_marginals = maxent.getMarginals(m_pwise)
data_marginals = maxent.getEmpiricalMarginals(train, m_pwise)

% get model weights
weights = maxent.getFactors(m_pwise);
writematrix(weights, 'ModelWeights/m_pwise.csv') 

% get explicit distribution
dist = maxent.getExplicitDistribution(m_pwise);
writematrix(dist, 'ModelWeights/m_pwise_dist_2.csv')

