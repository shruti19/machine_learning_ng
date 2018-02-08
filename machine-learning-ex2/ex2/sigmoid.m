function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
	g = zeros(size(z));
	m = length(z);
	g = 1 ./ (1 + e.^-z);
	% for i = 1:m
	% 	g(i) = 1/(1 + exp(-z(i)));
	% endfor
end
