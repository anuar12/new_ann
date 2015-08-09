function y = yGenerator(y, t, m, input_layer_size, num_labels, train_days)
% Creates a matrix (m-input_layer_size x input_layer_size) that can be 
% used for time-series computations from a linear matrix X.

    % Adds additional zeros for a column vector X
    y = [y zeros(t, input_layer_size - 1)];
    % Iterating through the number of training examples
    input_num = input_layer_size / num_labels;
    for j = 1:num_labels-1
        for i = 1:m 
            %part = y( i + 1 + j * train_days: i + num_labels + j * train_days - 1 );
            part = y( i + j * train_days );
            %size(X(i, (j*input_layer_size+2):(j*input_layer_size+input_layer_size)))
            y(i,j*input_num + 1) = part;
        end
    end
    % Erasing the last part of the matrix that is useless
    y = y(1:m, :);
end