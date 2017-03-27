function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
g = zeros(size(z));


gz = 1 ./(1 + exp(-z)); 

g = gz .* (1-gz);



end
