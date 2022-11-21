function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;    %12 x 2 (ex5 adds the ones column) * 2 x 1
sqerrors = (h - y).^2;
J1 = (1/(2*m)) * sum(sqerrors);

thetanew = theta(2:end);
sqtheta = thetanew.^2;
J2 = (lambda/(2*m))*sum(sqtheta);

J = J1 + J2;

%gradient
theta(1) = 0;

grad1 = (1/m) * X' * (h - y); 			% 2 x 12 * 12 x 1 = 2 x 1
grad2 = (lambda/m) * theta;

grad = grad1 + grad2;






% =========================================================================

grad = grad(:);

end
