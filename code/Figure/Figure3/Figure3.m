close all;
clc
clear

% Read Excel data
[data, txt, raw] = xlsread('Figure3.xlsx','sort');

% Determine data range
numPlots = 5;
plotsPerRow = 3;
rowsPerFigure = 2;

% Extract land cover type names
land_cover_types = {};
if size(txt, 1) >= 1
    land_cover_types = {'Forest', 'Grassland', 'Cropland', 'Wetland', 'Shrub'};
else
    land_cover_types = {'Type1', 'Type2', 'Type3', 'Type4', 'Type5'};
end

% Prepare bubble chart data
all_variables = {};
all_correlations = [];
all_scores = [];
all_types = {};
all_ranks = [];

txt(1,:) = [];

% Extract data for each land cover type
for i = 1:numPlots
    var_col = 3*i - 2;
    score_col = 3*i - 1;
    corr_col = 3*i - 2;
    
    % Get valid data (non-NaN)
    valid_idx = ~isnan(data(:, score_col)) & ~isnan(data(:, corr_col));
    valid_vars = txt(valid_idx, var_col);
    valid_scores = data(valid_idx, score_col);
    valid_corrs = data(valid_idx, corr_col);
    
    % Collect all data
    for j = 1:length(valid_vars)
        all_variables{end+1} = valid_vars{j};
        all_scores(end+1) = valid_scores(j);
        all_correlations(end+1) = valid_corrs(j);
        all_types{end+1} = land_cover_types{i};
        all_ranks(end+1) = j;
    end
end

% Calculate occurrence count for each variable
[unique_vars, ~, idx] = unique(all_variables);
var_counts = accumarray(idx, 1);

% Create mapping from variable to occurrence count
var_count_map = containers.Map(unique_vars, num2cell(var_counts));

% Get occurrence count for each data point
all_counts = zeros(size(all_variables));
for i = 1:length(all_variables)
    all_counts(i) = var_count_map(all_variables{i});
end

% Create bubble chart
figure;
set(gcf, 'Position', [100, 100, 1200, 800]);
set(gcf, 'Color', 'w');

% Assign colors to different land cover types
unique_types = unique(all_types);
colors = lines(length(unique_types));
type_colors = containers.Map();
for i = 1:length(unique_types)
    type_colors(unique_types{i}) = colors(i, :);
end

% Create bubble chart
hold on;
bubble_handles = [];
legend_labels = {};

for i = 1:length(unique_types)
    type_mask = strcmp(all_types, unique_types{i});
    if any(type_mask)
        bubble_sizes = all_counts(type_mask) * 50 + 20;
        
        h = scatter(all_correlations(type_mask), all_scores(type_mask), ...
                   bubble_sizes, ...
                   'filled', ...
                   'MarkerFaceColor', type_colors(unique_types{i}), ...
                   'MarkerEdgeColor', 'k', ...
                   'LineWidth', 1, ...
                   'DisplayName', unique_types{i});
        bubble_handles = [bubble_handles, h];
        legend_labels = [legend_labels, unique_types{i}];
    end
end

% Add variable name labels
font_size = 10;
for i = 1:length(all_variables)
    text(all_correlations(i) + 0.01, all_scores(i) + 0.01, ...
         sprintf('%s (%d)', all_variables{i}, all_counts(i)), ...
         'FontSize', font_size, ...
         'FontName', 'Arial', ...
         'HorizontalAlignment', 'left', ...
         'VerticalAlignment', 'bottom');
end

% Set axes and labels
xlabel('Correlation Coefficient', 'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('MRMR Score', 'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');

% Add legend
legend(bubble_handles, legend_labels, 'Location', 'bestoutside');
grid on;
box on;

% Set axis limits
xlim([0, 0.7]);
ylim([0,1.2]);

% Add bubble size legend
annotation('textbox', [0.85, 0.15, 0.12, 0.05], 'String', 'Bubble size ¡Ø Occurrence count', ...
           'FitBoxToText', 'on', 'BackgroundColor', 'none', 'FontSize', 10, 'FontWeight', 'bold');

hold off;

% Add statistics text box
stats_text = sprintf('Total variables: %d\nUnique variables: %d\nAverage correlation: %.3f\nAverage MRMR score: %.3f\nMax occurrence: %d', ...
                    length(all_variables), length(unique_vars), ...
                    mean(all_correlations), mean(all_scores), max(all_counts));
annotation('textbox', [0.15, 0.02, 0.25, 0.08], 'String', stats_text, ...
           'FitBoxToText', 'on', 'BackgroundColor', [0.9, 0.9, 0.9], ...
           'FontSize', 9, 'EdgeColor', 'none');

% Display occurrence count statistics
fprintf('Variable occurrence count statistics:\n');
for i = 1:length(unique_vars)
    fprintf('%s: %d times\n', unique_vars{i}, var_counts(i));
end

disp('Bubble chart created successfully! Bubble size determined by variable occurrence count.');