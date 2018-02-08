function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = length(theta);

[Jtmp, gradtmp] = costFunction(theta, X, y);
J = Jtmp + lambda/(2*m) * sum([0;theta(2:n, :)].^2);
% grad = gradtmp + lambda/m * gradtmp;

hx = sigmoid(X * theta);
% J = -1/m * [y' * log(hx) + (1 - y)' * log(1 - hx)] + lambda/(2*m) * sum(theta.^2);
grad = 1/m * (X' * (hx - y)) + lambda/m * theta;
grad(1) = gradtmp(1);

end
