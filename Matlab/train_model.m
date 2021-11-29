%% load data
clear
load data2014

%% explore data
[total_N, obvs] = size(train_reps);

% select different subsets of the data and compare properties
NN = 100;  % number of neurons
rep = 1; % seed for random number generator

rng(rep)
id_N = randperm(total_N, NN); % id of neurons

subset = train_reps(id_N,:); % select the training data


%% Fitting models
% We need to specify a model name 'name', number of neurons 'NN', which repeat of the
% train this is 'rep'(defining which subset we train on), and whether we
% use data shuffled over time or shuffled over stimulus 'shuffle'. 
% We then need to save the matlab models, and the model weights as csv
% files. We will name the files `shuffle_NN_name_rep`

shuffle = 'stimulus'; % 'stimulus' or 'time'
name = 'pairwise'; % 'indep' 'ksync' 'pairwise' or 'kpairwise'
NN = 100;  % 10  40 (70) 100 (130) (160)
rep = 2; % also use this to set the seed of the random number generator
file_name = "TrainedModels/" + shuffle + "_" + NN + "_" + name + "_" + rep;

% select which neurons activities to train model on
rng(rep) % set seed based on rep - is this a good idea to do?
id_N = randperm(total_N, NN); % id of neurons activity we are going to train on

train = train_reps(id_N,:); % select the training data

model = maxent.createModel(NN, name); % declare the model

if strcmp(name,'ksync')
    model = get_Ksync(train);
else
    model = maxent.trainModel(model, train, 'threshold', 1, 'savefile', 'checkpoint'); %train model while saving to file
end
    
% save model weights and save model
save(file_name, 'model'); % save model
weights = maxent.getFactors(model); % get model weights
writematrix(weights, file_name+'.csv'); % save model weights 