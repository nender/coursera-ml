function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
tt = theta';

acc = 0;
for i = 1:m
  hypothesis = tt * X(i, :)';
  sum = (hypothesis - y(i))^2;
  acc = acc + sum;
end
acc = acc /( 2*m);

% You need to return the following variables correctly 
J = acc;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
