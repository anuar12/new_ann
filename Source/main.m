function [historyJ, X, y] = main(input_layer_size, hidden_layer_size, lambda)
%clear ; 
%close all; clc

%% Setup the parameters you will use for this exercise
%input_layer_size  = 120 * 5;  % # of input variables
%hidden_layer_size = 85 * 5;     % # of hidden units
num_labels = 5;           % # of classes
train_days = 3662;
test_days = 252;

%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('inputs.mat');
X = [Open(1:train_days); Close(1:train_days); High(1:train_days);...
                           Low(1:train_days); Volume(1:train_days)];
                       
t = size(X,1);                % # of total points in X
m = train_days - input_layer_size;  % # of training examples for backprop        
d = train_days / input_layer_size;  % # of separate individual training intervals

X = XGenerator(X, t, m, input_layer_size, num_labels, train_days);

y = [ Open(input_layer_size + 1:train_days);...
      Close(input_layer_size + 1:train_days);...
      High(input_layer_size + 1:train_days);...
      Low(input_layer_size + 1:train_days);...
      Volume(input_layer_size + 1:train_days); ];
  
% max and min values of every parameter are computed and stored so that we
% can rescale the output of the neural network to original scale.
maxy1 = max(y(1:m));        miny1 = min(y(1:m));
maxy2 = max(y(m+1:2*m));    miny2 = min(y(m+1:2*m));
maxy3 = max(y(2*m+1:3*m));  miny3 = min(y(2*m+1:3*m));
maxy4 = max(y(3*m+1:4*m));  miny4 = min(y(3*m+1:4*m));
maxy5 = max(y(4*m+1:5*m));  miny5 = min(y(4*m+1:5*m));
save('maxmin.mat', 'maxy1', 'miny1', 'maxy2', 'miny2', 'maxy3', 'miny3',...
                   'maxy4', 'miny4', 'maxy5', 'miny5');

% Scale every input parameter in y to fit [0,1], since sigmoid function
% outputs values in range [0,1].
y(1:m) = ( y(1:m) - min(y(1:m)) ) / ( max(y(1:m)) - min(y(1:m)) );    
y(m+1:2*m) = ( y(m+1:2*m) - min(y(m+1:2*m)) ) / ( max(y(m+1:2*m)) - min(y(m+1:2*m)) );
y(2*m+1:3*m) = ( y(2*m+1:3*m) - min(y(2*m+1:3*m)) ) / ( max(y(2*m+1:3*m)) - min(y(2*m+1:3*m)) );
y(3*m+1:4*m) = ( y(3*m+1:4*m) - min(y(3*m+1:4*m)) ) / ( max(y(3*m+1:4*m)) - min(y(3*m+1:4*m)) );
y(4*m+1:5*m) = ( y(4*m+1:5*m) - min(y(4*m+1:5*m)) ) / ( max(y(4*m+1:5*m)) - min(y(4*m+1:5*m)) );

y = yGenerator(y, 5*m, m, 5, num_labels, train_days);

% Random 300 points are selected for visualisation
fprintf('Visualising 300 random points of Open price.\n');
figure(100);
rand_num = randi([1 1500]);
plot([rand_num:rand_num+299], X(rand_num:rand_num+299,1));
title('Open Price vs Time', 'FontSize', 16);
xlabel('Time (days)', 'FontSize', 14);
ylabel('Open Price (USD)', 'FontSize', 14);

% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ================ Part 6: Initializing Parameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n');

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
J = nnCostFunction2(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf('Cost at initial parameters : %f \n', J);

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n');

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
MaxIter = 35000;
options = optimset('MaxIter', MaxIter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost, i, historyJ] = fmincg(costFunction, initial_nn_params, options);

J = nnCostFunction2(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf('Cost at final parameters : %f \n', J);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
save('thetas.mat', 'Theta1', 'Theta2');

historyJ(historyJ==0.0) = [];
historyJ = reshape(historyJ, MaxIter, 2);
figure(101);
plot(historyJ(:,1), historyJ(:,2));
grid on;
title_name = sprintf( 'Cost Function vs. Iteration' );
title(title_name);
xlabel('Iteration');
ylabel('Cost Function J');
%testX(X, y);
             
% fprintf('Program paused. Press enter to continue.\n');
% pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

%pred = predict(Theta1, Theta2, X);

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
end

