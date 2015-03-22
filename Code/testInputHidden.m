function testInputHidden(input_layer_sizes, hidden_layer_sizes)
% Test the neural network with various sizes of input layer and hidden
% layer. The figures of (Cost Function vs Iteration) are plotted for each
% input layer size that contain several plots of different hidden layer
% sizes. 
    close all;
    colors = ['b' 'g' 'r' 'c' 'm' 'y' 'k'];
    colors2 = [ [0.3 0.6 0.9] [0.8 0.3 0.4] [0.3 0.8 0.2] ];
    legend_str = [];
    line = zeros( 1, numel(hidden_layer_sizes) ); 
    
    % Iterate through every value of input layer sizes
    for i = 1:numel(input_layer_sizes)
        fig_handle = figure(i);
        layer_size = input_layer_sizes(i);
        
        % Iterate through every value of hidden layer sizes and train the
        % network for a corresponding pair of (input_layer, hidden_layer).
        for j = 1:numel(hidden_layer_sizes)
            hidden_size = hidden_layer_sizes(j);
            historyJ = main(layer_size, hidden_size, 0.1);
            figure(fig_handle);
            if j > 7
                color = colors2(j-7:j-5);
                line(j) = plot( historyJ(:,1), historyJ(:,2), 'Color', color );
            else
                color = colors(j);
                line(j) = plot( historyJ(:,1), historyJ(:,2), color );
            end
            hold on;
            str = cellstr(sprintf('Hidden = %d ', hidden_size));
            legend_str = [legend_str str];
        end
        legend(line, legend_str, 'Location', 'North');  
        grid on;
        title_name = sprintf('Cost Function vs. Iteration (inp size=%d)', layer_size);
        title(title_name, 'FontSize', 16);
        xlabel('Iteration', 'FontSize', 14);
        ylabel('Cost Function J', 'FontSize', 14);
    end
end