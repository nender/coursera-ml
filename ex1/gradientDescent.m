function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
features = 2;
J_history = zeros(num_iters, 1);

function t = GDTerm(j)
  accumulator = 0;
  
  for i = 1:m
    xterm = X(i, :);
    hypothesis = xterm * theta;
    accumulator += (hypothesis - y(i)) * xterm(j);
  end
  
  t = (alpha / m) * accumulator;
end

for iter = 1:num_iters
    t1 = GDTerm(1);
    t2 = GDTerm(2);
    theta(1) -= t1;
    theta(2) -= t2;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
