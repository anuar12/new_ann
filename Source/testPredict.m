function testPredict(testingX, y, n_predict)
    figure(2);
    load('thetas.mat');
    load('inputs.mat');
    
    % Feedforwarding through the neural net with computed weights.
    [p, h2] = predict(Theta1, Theta2, testingX);
    % Rescaling back to original
    h2 = ( h2 .* ( maxy - miny ) ) + miny;
    
    for i = 1:n_predict
        testingX = [testingX(1, :)'; h2];
        m = size(testingX, 1);
        testingX = XGenerator(testingX, m, m-600+1, 600);
        [p, adding] = predict(Theta1, Theta2, testingX);
        adding = ( adding(end) .* ( maxy - miny ) ) + miny;
        h2 = [h2; adding];
    end
    plot([1:n_predict+1], h2(1:end));
    hold on;
    plot([1:n_predict+1], X(1+600:601+n_predict), 'g');
end