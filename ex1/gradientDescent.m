function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    newTheta = 0;
    for i=1:m,
        newTheta = newTheta + alpha/m * (theta' * X(i,:)' - y(i)) * X(i,:)';
    end
    theta = theta - newTheta;  
    J_history(iter) = computeCost(X, y, theta);

end;

end
