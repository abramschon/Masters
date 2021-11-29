%% load a subset of 10 neurons from the data
clear
load data2014

NN = 5;
train = train_rand(1:NN,:); % select activity of the first 10 neurons
test = test_rand(1:NN,:);

% create a model and train as usual

m_ksync = maxent.createModel(NN,'ksync'); %create model
m_ksync = maxent.trainModel(m_ksync, train); %train model

% now an analytic solution

m2_ksync = get_Ksync(train);

% compare weights and distributions
v_k = maxent.getFactors(m2_ksync) 
v_k_2 = maxent.getFactors(m_ksync)
figure
scatter(v_k,v_k_2);
hold on;
xlabel('Our weights');
ylabel('Their weights');
title('Comparison of Weights');
hold off;

our_dist = maxent.getExplicitDistribution(m2_ksync);
their_dist = maxent.getExplicitDistribution(m_ksync);
figure
loglog(our_dist,their_dist,'o');
hold on;
xlabel('Our dist');
ylabel('Their dist');
title('Comparison of distribution');

actual_dist = maxent.getEmpiricalModel(train);
figure
scatter(log(our_dist(1:28)),actual_dist.logprobs);
hold on;
xlabel('Our dist');
ylabel('Actual dist');
title('Comparison to true dist');
