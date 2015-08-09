function testX(X)
    load('thetas.mat');
    load('inputs.mat');
    load('maxmin.mat');
    
    m = size(X, 1);
    input_layer_size = size(X, 2);
    % Feedforwarding through the neural net with computed weights.
    [p, h2] = predict(Theta1, Theta2, X);
    % Rescaling back to original
    h2(:,1) = ( h2(:,1) .* ( maxy1 - miny1 ) ) + miny1;
    h2(:,2) = ( h2(:,2) .* ( maxy2 - miny2 ) ) + miny2;
    h2(:,3) = ( h2(:,3) .* ( maxy3 - miny3 ) ) + miny3;
    h2(:,4) = ( h2(:,4) .* ( maxy4 - miny4 ) ) + miny4;
    h2(:,5) = ( h2(:,5) .* ( maxy5 - miny5 ) ) + miny5;

    y = [ Open(input_layer_size + 1:m);...
      Close(input_layer_size + 1:m);...
      High(input_layer_size + 1:m);...
      Low(input_layer_size + 1:m);...
      Volume(input_layer_size + 1:m); ];
    size(y)
    figure(3);
    line1 = plot([1:100], h2(1:100,1));
    str1 = 'Prediction';
    hold on;
    line2 = plot([1:100], y(1:100), 'g');
    str2 = cellstr('Output from the set');
    legend([line1 line2], [str1 str2]);
    title('Open Price vs Time', 'FontSize', 16);
    xlabel('Time (days)', 'FontSize', 14);
    ylabel('Open Price (USD)', 'FontSize', 14);
    figure(4);
    line3 = plot([1:100], h2(1:100,5), 'k');
    str3 = 'Prediction';
    hold on;
    line4 = plot([1:100], y(m*4+1:m*4+100), 'g');
    str4 = cellstr('Output from the set');
    legend([line3 line4], [str3 str4]);
    title('Volume vs Time', 'FontSize', 16);
    xlabel('Time (days)', 'FontSize', 14);
    ylabel('Volume', 'FontSize', 14);
end