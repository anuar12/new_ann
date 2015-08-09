function testLambda(lambdas)
% Test the neural network with various values of learning rates, lambda. 
    close all;
    colors = ['b' 'g' 'r' 'c' 'm' 'k'];
    legend_handle = [];
    fig_handle = figure(20);
    line = zeros( 1, numel(lambdas) ); 
    legend_str = [];
    % Iterate through the values of lambdas and train the neural network
    % for each of them. 
    for i = 1:numel(lambdas)
        lambda = lambdas(i);
        historyJ = main(600, 425, lambda);
        color = colors(i);
        figure(fig_handle);
        line(i) = plot( historyJ(:,1), historyJ(:,2), color );
        hold on;
        str = cellstr(sprintf('lambda %.1f ', lambda));
        legend_str = [legend_str str];
    end
    grid on;
    legend(line, legend_str, 'Location', 'North');
    title('Cost Function vs. Iteration', 'FontSize', 16);
    xlabel('Iteration', 'FontSize', 14);
    ylabel('Cost Function J', 'FontSize', 14);
end