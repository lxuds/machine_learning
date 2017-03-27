function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function and gradient

%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%




J =  sum(sum(((X * Theta' - Y) .* R) .^2))/2 + lambda/2 * sum(sum(Theta.^2,1)) + lambda/2 * sum(sum(X.^2,1)); 

X_grad = (X * Theta'.*R - Y.*R) * Theta + lambda * X; % Nm x Nu * Nu x n = Nm x n

Theta_grad = (X * Theta'.*R - Y.*R)' * X + lambda * Theta;  % Nu x Nm * Nm x n = Nu x n 








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
