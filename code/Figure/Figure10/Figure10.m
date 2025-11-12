clear
clc
close all

% Read Excel data
data = readmatrix('Figure10.xlsx'); % Replace 'your_file.xlsx' with your actual filename

% Extract data
years = data(:, 1); % First column: years
nep_data = data(:, 2:7); % Columns 2-7: six NEP product data

% Calculate average of six products
avg_nep = mean(nep_data, 2); % Calculate mean by row

% Create figure
figure('Position', [100, 100, 1000, 600]);

% Define colors and line styles (ensure six lines have distinct differences)
colors = lines(6); % Use MATLAB's lines colormap
line_styles = {'-', '--', ':', '-.', '-', '--'};
markers = {'o', 's', 'd', '^', 'v', '>'};

% Plot line charts for six products
hold on;
for i = 1:6
    plot(years, nep_data(:, i), ...
        'Color', colors(i, :), ...
        'LineStyle', line_styles{i}, ...
        'Marker', markers{i}, ...
        'MarkerSize', 5, ...
        'LineWidth', 2, ...
        'DisplayName', sprintf('Product %d', i));
end

% Plot average black line (thick line, no markers)
plot(years, avg_nep, ...
    'Color', 'k', ...
    'LineStyle', '-', ...
    'LineWidth', 3, ...
    'Marker', 'none', ...
    'DisplayName', 'Ensemble Mean');

hold off;

% Set graph properties
xlabel('Year', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('NEP Value', 'FontSize', 12, 'FontWeight', 'bold');
grid on;

% Add legend
legend('Location', 'best', 'FontSize', 10);

% Set x-axis ticks as integer years (if years are integers)
if all(round(years) == years)
    set(gca, 'XTick', unique(years));
    xtickangle(45); % Rotate x-axis labels to avoid overlap
end

% Beautify graph
set(gca, 'FontSize', 10, 'LineWidth', 1.5);
box on;

% Automatically adjust y-axis range to include all data
ylim([min(nep_data(:)) * 0.95, max(nep_data(:)) * 1.05]);