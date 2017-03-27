%% Machine Learning  - Neural Network Learning

%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%% Initialization

%% Setup the parameters 
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that  mapped "0" to label 10)

%% =========== Part 0: Loading and Visualizing Data =============
%  loading and visualizing the dataset. 
%   dataset  contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));


%% ================ Part 1: Initializing Pameters ================
%  implment a two layer neural network that classifies digits. 
%  Frist implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%%================= Part 2: Training NN ===================
%  Train a neural network. 
%  Use an advanced optimizer "fmincg" for the gradient computations
%
fprintf('\nTraining Neural Network... \n')

%  Could change the MaxIter to a larger value to see how more training helps.
options = optimset('MaxIter', 50);

%  Could try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%%===================Part 3: Visualize Weights =================
%   "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));


%%===================Part 4: Implement Predict =================
% implement the "predict" function to use the
%  neural network to predict the labels of the training set

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


