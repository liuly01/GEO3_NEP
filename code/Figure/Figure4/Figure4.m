clc;
clear;
close all;

% Set folder path
folderPath = 'GEO3_anlysis\Result';

% Get all xlsx files in the folder
fileList = dir(fullfile(folderPath, '*.xlsx'));

% Create figure window
figure;
set(gcf, 'Position', [100, 100, 1200, 900]); % Set larger figure window

% Define row numbers to plot and corresponding subplot titles
rowNumbers = [7, 16, 25, 34];
titles = {'1', '2', '3', '4'};

% Pre-load all color values to determine global range
allColorValues = [];
for i = 1:length(fileList)
    filePath = fullfile(folderPath, fileList(i).name);
    for row = 43
        colorValue = xlsread(filePath, 1, ['B' num2str(row) ':K' num2str(row)]);
        allColorValues = [allColorValues, colorValue];
    end
end
minColor = min(allColorValues);
maxColor = max(allColorValues);

% Create 2¡Á2 subplots
for plotIdx = 1:4
    subplot(2, 2, plotIdx);
    hold on;
    
    % Iterate through each file
    for i = 1:length(fileList)
        fileName = fileList(i).name;
        filePath = fullfile(folderPath, fileName);
        
        % Read current row data as y values
        y = xlsread(filePath, 1, ['B' num2str(rowNumbers(plotIdx)) ':K' num2str(rowNumbers(plotIdx))]);
        
        % Read row 43 data as color values (all subplots use same color mapping)
        colorValue = xlsread(filePath, 1, 'B43:K43');
        
        % Determine marker shape based on filename
        if contains(fileName, 'forest', 'IgnoreCase', true)
            marker = 'p';  % pentagram
        elseif contains(fileName, 'grassland', 'IgnoreCase', true)
            marker = 's';  % square
        elseif contains(fileName, 'cropland', 'IgnoreCase', true)
            marker = 'o';  % circle
        elseif contains(fileName, 'wetland', 'IgnoreCase', true)
            marker = '^';  % upward triangle
        elseif contains(fileName, 'shrub', 'IgnoreCase', true)
            marker = 'd';  % diamond
        else
            marker = '.';  % default point
        end
        
        % Plot scatter points
        scatter(1:10, y, 60, colorValue, 'filled', 'Marker', marker);
    end
    
    % Subplot decoration
    xlabel('X Value (1-10)', 'FontSize', 10);
    ylabel('Y Value', 'FontSize', 10);
    title(titles{plotIdx}, 'FontSize', 12);
    grid on;
    
    % Add legend to first subplot only
    if plotIdx == 1
        legendLabels = {};
        markers = {};
        if any(contains({fileList.name}, 'forest', 'IgnoreCase', true))
            legendLabels{end+1} = 'Forest';
            markers{end+1} = 'p';
        end
        if any(contains({fileList.name}, 'grassland', 'IgnoreCase', true))
            legendLabels{end+1} = 'Grassland';
            markers{end+1} = 's';
        end
        if any(contains({fileList.name}, 'cropland', 'IgnoreCase', true))
            legendLabels{end+1} = 'Cropland';
            markers{end+1} = 'o';
        end
        if any(contains({fileList.name}, 'wetland', 'IgnoreCase', true))
            legendLabels{end+1} = 'Wetland';
            markers{end+1} = '^';
        end
        if any(contains({fileList.name}, 'shrub', 'IgnoreCase', true))
            legendLabels{end+1} = 'Shrub';
            markers{end+1} = 'd';
        end
        
        % Create custom legend
        h = zeros(length(legendLabels), 1);
        for k = 1:length(legendLabels)
            h(k) = plot(NaN, NaN, markers{k}, 'MarkerSize', 8, 'LineWidth', 1.5, 'Color', 'k');
        end
        legend(h, legendLabels, 'Location', 'northeast', 'FontSize', 9);
    end
    
    hold off;
end

% Add global colorbar
colormap(parula);
caxis([minColor, maxColor]);
c = colorbar('Position', [0.92 0.15 0.02 0.7]); % Adjust colorbar position and size
c.Label.String = 'Color Value';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';

% Adjust subplot spacing
set(gcf, 'Color', 'w');
ha = findobj(gcf, 'type', 'axes');
for i = 1:length(ha)
    pos = get(ha(i), 'Position');
    set(ha(i), 'Position', [pos(1)*0.9, pos(2), pos(3)*0.85, pos(4)]);
end