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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% logarithmic term of J 
h = sigmoid(X * theta);
log1 = y' * log(h);
log2 = (1-y)' * log(1-h);
J1=(-1/m)*(log1 + log2);

% term from J related with theta

%theta needs to be shortened (theta0 needs to be out for the formula)
thetashort = theta(2:length(theta),1);
sumsqr_theta = thetashort' * thetashort;
J2 = (lambda/(2*m)) * sumsqr_theta;
%cost function			
J = J1 + J2;

dif = h - y;
%gradient
for iter = 1:length(theta)
	if (iter == 1)
		grad(iter,1) = (1/m) * (dif' * X(:,iter));
	else
		grad(iter,1) = 	((1/m) * dif' * X(:,iter)) + ((lambda/m) * theta(iter,1));

	endif
end	

% =============================================================

end
