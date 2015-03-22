function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, m)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification.
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m, 1) X];   % Adding bias unit.  m x (input_layer_size + 1)
z2 = a1 * Theta1';     % m x hidden_layer_size
a2 = sigmoid(z2);      % Computing a hidden layer
a2 = [ones(m, 1) a2];  
z3 = a2 * Theta2';     
% Computing final layer, which is described by probability of occurence of
% certain class (# of training examples x # of classes)
htheta = sigmoid(z3);  % Final output

% Computing cost function.
% Enumerating through the training examples
for i = 1:m
    hthetai = htheta(i, :);
    yi = y(i);
    Ji = (-1 / m) * (yi*log(hthetai') + (1-yi) * log(1-hthetai'));
    %Ji = (1/2)*(hthetai - yi)^2;
    J = J + Ji;
end

% regularization = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2)));
% J = J + regularization;


% Computing error terms (delta) for every unit for each of the training
% example. First, by computing delta3 (output layer) with respect to y and
% then backpropagating it to hidden layer (delta2).
Delta2 = 0;
Delta1 = 0;
for i = 1:m
    hthetai = htheta(i, :);   % 1 x num_labels
    yi = y(i);                % 1 x num_labels
    delta3 = (hthetai - yi);    % 1 x num_labels
    delta2 = Theta2' * delta3' .* sigmoidGradient([1 z2(i, :)])';  % hidden_layer_size + 1 x 1
    delta2 = delta2(2:end);      % getting rid of bias unit
    
    Delta2 = Delta2 + delta3' * a2(i, :);   % num_labels x hidden_layer_size + 1
    Delta1 = Delta1 + delta2 * a1(i, :);    % hidden_layer_size x input_layer_size + 1
end

Theta1_grad = (1/m) * Delta1 + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = (1/m) * Delta2 + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
