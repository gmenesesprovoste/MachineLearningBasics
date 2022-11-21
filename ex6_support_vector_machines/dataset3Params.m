function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C = 1;
sigma = 0.1;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


k = 4;
values = zeros(k*2,1);

%implementing all combinations of C and sigma values. Limited case for (2*k)^2 combinations
% and multiples of 0.01 and 0.03
V1 = zeros(k,1);
V2 = zeros(k,1);

v1 = 0.01;
v2 = v1*3;
for i = 1:k
	V1(i,1) = v1 * 10^(i-1);
	V2(i,1) = v2 * 10^(i-1);
end	
V = [V1;V2];
[a b] = ndgrid(V);
allcom = [a(:),b(:)];

%loop for calculating error for different C and sigma combinations. E must be initialized as a large number  
E = 1000;
for j = 1:size(allcom,1)
	C_test = allcom(j,1);
	sigma_test = allcom(j,2);
	model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
	predictions = svmPredict(model,Xval);
	err = mean(double(predictions ~= yval));
	if err < E
		E = err;
		C = C_test;
		sigma=sigma_test;
	end	

end


% =========================================================================
end
