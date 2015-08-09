function X = XGenerator(X, t, m, input_layer_size, num_labels, train_days)
% Creates a matrix (m-input_layer_size x input_layer_size) that can be 
% used for time-series computations from a linear matrix X.

    % Adds additional zeros for a column vector X
    X = [X zeros(t, input_layer_size - 1)];
    % Iterating through the number of training examples
    input_num = input_layer_size / num_labels;
    for j = 0:num_labels-1
        for i = 1:m
            if j == 0
                part = X( i + 1 + j * train_days: i + input_num + j * train_days - 1 );
                X(i,2:input_num) = part;
            else
                part = X( i + 1 + j * train_days: i + input_num + j * train_days );
                %size(X(i, (j*input_layer_size+2):(j*input_layer_size+input_layer_size)))
                X(i,j*input_num+1:j*input_num+input_num) = part;
        end
    end
    % Erasing the last part of the matrix that is useless
    X = X(1:m, :);
end