clear
clc
close all

% Read data
data = xlsread('Figure5.xlsx', 'Uncertainty');
data_subset = data(2:25, :);
years = data_subset(:, 1);
NEP_monthly = data_subset(:, 2:13);

% Create time series
num_months = size(NEP_monthly, 1) * size(NEP_monthly, 2);
time_points = linspace(years(1), years(end) + 11/12, num_months);
NEP_flat = reshape(NEP_monthly', 1, []);

% Plot
figure;
plot(time_points, NEP_flat, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'Color', 'b');
xlabel('YEAR', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('NEP', 'FontSize', 12, 'FontWeight', 'bold');
title('2001-2024 NEP', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
hold on;

% Get y-axis range first
yl = ylim;  % Get current y-axis range
ylim_min = yl(1)+0.01;
ylim_max = yl(2);

% Set light gray background for odd years
for year = years(1):years(end)
    if mod(year, 2) == 1  % Odd years
        % Draw background rectangle
        rectangle('Position', [year, ylim_min, 1, ylim_max-ylim_min], ...
                 'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none');
    end
end

% Re-plot data points to ensure they are on top of background
plot(time_points, NEP_flat, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
    'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'Color', 'b');

% Set x-axis - year labels at middle of each year
year_centers = years(1):years(end);
x_tick_positions = year_centers + 0.5;

set(gca, 'XTick', x_tick_positions);
set(gca, 'XTickLabel', arrayfun(@num2str, year_centers, 'UniformOutput', false));
xlim([years(1), years(end) + 1]);

% Add legend
legend('NEP Monthly Values', 'Location', 'best');

% Beautify plot
set(gca, 'FontSize', 10, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');