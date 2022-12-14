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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% for this exercise: theta1 - 25 x 401, theta2 - 10x26, X - 5000 x 400 (ones need to be added)
Xones = [ones(m,1) X];
a1 = Xones; %5000 x 401
z2 = a1 * Theta1';% 5000 x 25 = 5000 x 401 * 401 * 25 -- m x h
%z2pre = X * Theta1(:,2:end)'; 
a2 = sigmoid(z2);%5000 x 25
a2ones = [ones(m,1) a2];%5000 x 26
z3 = a2ones*Theta2';
h = sigmoid(z3);
%size(h)
%size(y)

%enlarging vector y to 5000 x 10, with ones and zeros
ylarge = zeros(m,num_labels);
for iter = 1:m
	ylarge(iter,y(iter)) = 1;
end	
%ylarge	
%size(ylarge)

for k = 1:num_labels
	yk = ylarge(:,k);	%5000 x 1
	hk = h(:,k);	%5000 x 1
	log1 = yk' * log(hk);
	log2 = (1-yk') * log(1-hk);
	J = J + log1 + log2;
end
%regularization
Th12 = Theta1(:,2:end) .* Theta1(:,2:end);
Th22 = Theta2(:,2:end) .* Theta2(:,2:end);
Th1cuad=sum(Th12(:));
Th2cuad=sum(Th22(:));
reg = (lambda/(2*m)) * (Th1cuad + Th2cuad);
 

J = (-1/m) * J + reg;

% ==================grad =======================================================

d3 = h - ylarge; %5000 x 10

%if I use the approximation for the sigmoidgradient, I have to remove the first column of d2 when I compute delta
%d2 = d3 * Theta2(:,2:end) .* z2 .* (1-z2); % 5000 x 10 * 10 x 25 --- 5000 x 25 (z2 dimensions)
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

delta1 = d2' * a1; %  25 x 5000 * 5000 x 401 ---25 x 401
delta2 = d3' * a2ones; % 10 x 5000 * 5000 x 25

%regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;


Theta1_grad = (1/m) * delta1 + ((lambda/m) * Theta1);
Theta2_grad = (1/m) * delta2 + ((lambda/m) * Theta2);

% Unroll 
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
