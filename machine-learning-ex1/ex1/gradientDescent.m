function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % theta0 = theta0 - alpha * 1/m * sum(theta0 + theta1*x - y)
    % theta1 = theta1 - alpha * 1/m * sum((theta0 + theta1*x - y) * x)
    % X and theta are mx2 and 2x1 vectors. Their product results in sum(theta0 + theta1*x - y) expr above
    
    theta0 = theta(1) - alpha * 1/m * sum(X * theta - y); 
    theta1 = theta(2) - alpha * 1/m * sum((X * theta - y) .* X(:, 2));  %(X * theta - y)' * X(:,1);
    theta = [theta0; theta1];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %fprintf('J_history(%d) = %f\n', iter, J_history(iter))

end

end

