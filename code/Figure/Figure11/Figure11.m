close all;
clc
clear

% Read Excel data
filename = 'Figure11.xlsx'; % Replace with your actual filename
data = readmatrix(filename, 'Range', 'A1:Y71'); % Assuming data is in range A1:Y71

% Define region names (modify according to your actual situation)
region_names = {
    'North America', 'South America', 'Russia', 'Europe', ...
    'West Asia', 'Africa', 'East Asia', 'South Asia', ...
    'Southeast Asia', 'Oceania'
};

% Define product names (modify according to your actual situation)
product_names = {'GEO3', 'FLUXCOM-XBASE', 'Zeng et al., 2020', 'Huang et al., 2021', 'Han et al., 2025', 'Gloflux'};

% Extract years (from first row of data)
years = data(1, 2:end); % Assuming first row from column 2 contains years

% Initialize cell array to store region data
region_data = cell(10, 6); % 10 regions, 6 products

% Extract data for each region
for product_idx = 1:6
    start_row = (product_idx - 1) * 12 + 2; % Starting row for each product data
    for region_idx = 1:10
        row_idx = start_row + region_idx - 1;
        region_data{region_idx, product_idx} = data(row_idx, 2:end);
    end
end

% Calculate average of six products for each region
region_avg = cell(10, 1);
for region_idx = 1:10
    % Combine data from six products into matrix
    all_data = zeros(6, length(years));
    for product_idx = 1:6
        all_data(product_idx, :) = region_data{region_idx, product_idx};
    end
    % Calculate mean
    region_avg{region_idx} = mean(all_data, 1);
end

% Create figure window
figure('Position', [100, 100, 1600, 1000], 'Name', 'RECCAP Region NEP Value Comparison');

% Define colors and line styles
colors = lines(6); % 6 different colors
line_styles = {'-', '--', ':', '-.', '-', '--'};
markers = {'o', 's', 'd', '^', 'v', '>'};

% Plot 10 subplots
for region_idx = 1:10
    subplot(3, 4, region_idx);
    hold on;
    
    % Plot data for six products in this region
    for product_idx = 1:6
        plot(years, region_data{region_idx, product_idx}, ...
            'Color', colors(product_idx, :), ...
            'LineStyle', line_styles{product_idx}, ...
            'Marker', markers{product_idx}, ...
            'MarkerSize', 4, ...
            'LineWidth', 1.5, ...
            'DisplayName', product_names{product_idx});
    end
    
    % Plot average of six products (thick black line)
    plot(years, region_avg{region_idx}, ...
        'Color', 'k', ...
        'LineStyle', '-', ...
        'LineWidth', 3, ...
        'Marker', 'none', ...
        'DisplayName', 'Ensemble Mean');
    
    % Set subplot properties
    title(region_names{region_idx}, 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Year', 'FontSize', 8);
    ylabel('NEP (PgC)', 'FontSize', 8);
    grid on;
    box on;
    
    % Set x-axis ticks
    set(gca, 'XTick', years(1:2:end), 'FontSize', 7);
    xtickangle(45);
    
    % Add legend to first subplot only
    if region_idx == 1
        legend('Location', 'best', 'FontSize', 7);
    end
    
    hold off;
end

% Adjust subplot positions to center the two plots in the third row
subplot_handles = get(gcf, 'Children');
subplot_handles = flipud(subplot_handles); % Reverse handle order