function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%  Implement regularization with the cost function and gradients.




X = [ones(m, 1) X];
a1 = X;
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');

%[p_value, p_index] = max(hTheta, [], 2);


%p = p_index;

%p = p(:);

yy = zeros(m, num_labels);
%yy(:, y(:,1)) = 1
for i = 1:m
   K_y = y(i,1);
   yy(i,K_y) = 1;
end;

J = sum(sum( - yy .* log(a3) - (1 - yy) .* log(1-a3))) / m + lambda/(2*m)* sum((sum(Theta1(:,2:end).^2))) ...
          + lambda/(2*m)* sum((sum(Theta2(:,2:end).^2)));


%===================

delta3 = a3 - yy;
delta2 = delta3 * Theta2(:,2:end) .* a2(:,2:end) .* (1-a2(:,2:end));


temp1 = Theta1;
temp1(:,1) = 0;


temp2 = Theta2;
temp2(:,1) = 0;


Theta1_grad = Theta1_grad +  delta2' * a1/m  + lambda * temp1/m; 
Theta2_grad = Theta2_grad +  delta3' * a2/m + lambda * temp2/m;;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
