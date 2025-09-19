clc;
clear;
close all;

% Define paths
landcover_path = 'D:\han05\variables\MODIS\MM\IGBP\0degree05\';
variables_selection_file = 'D:\han05\data\fluxdata\MM\Classification\PFT\PFT_select_variables.xlsx';
predictive_vars_path = 'D:\han05\variables\predictive\';
sample_files_path = 'D:\han05\data\fluxdata\MM\Classification\PFT\dataset_newevi\';
output_path = 'D:\han05\GEO3_NEE\';

% Read variable selection table
var_selection = readtable(variables_selection_file);
pft_types = var_selection.PFT;

% Define land cover type classification
landcover_classes = {
    [1,2,3,4,5], 'forest';
    [6,7], 'shrub';
    [8,9,10], 'grassland';
    [11], 'wetland';
    [12], 'cropland';
    % Others will be ignored
};

% Process each year
for year = 2001:2024  % Adjust the range as needed
    % Read land cover data for the year
    lc_file = dir(fullfile(landcover_path, sprintf('*%d*.tif', year)));
    [landcover, R] = geotiffread(fullfile(landcover_path, lc_file(1).name));
    
    % Create output directory for the year
    year_output_path = fullfile(output_path, num2str(year));
    if ~exist(year_output_path, 'dir')
        mkdir(year_output_path);
    end
    
    % Process each month
    for month = 1:12
        fprintf('Processing year %d, month %d...\n', year, month);
        
        % Initialize NEE result matrix
        NEE = nan(size(landcover));
        NEE_uncertainty = nan(size(landcover));
        
        % Pre-process landcover mask to identify pixels we need to process
        valid_pixels = false(size(landcover));
        for lc_idx = 1:size(landcover_classes, 1)
            valid_pixels = valid_pixels | ismember(landcover, landcover_classes{lc_idx, 1});
        end
        valid_pixels = valid_pixels & ~isnan(landcover);
        
        % Get linear indices of all valid pixels
        [valid_rows, valid_cols] = find(valid_pixels);
        num_valid_pixels = length(valid_rows);
        
        % Process each land cover type
        for lc_idx = 1:size(landcover_classes, 1)
            lc_values = landcover_classes{lc_idx, 1};
            lc_name = landcover_classes{lc_idx, 2};
            
            % Find pixels with this land cover type
            lc_mask = ismember(landcover, lc_values) & valid_pixels;
            if ~any(lc_mask(:))
                continue;
            end
            
            % Get linear indices of pixels for this land cover type
            [lc_rows, lc_cols] = find(lc_mask);
            num_lc_pixels = length(lc_rows);
            
            % Get required variables for this land cover type
            pft_row = strcmpi(pft_types, lc_name);
            req_vars = var_selection{pft_row, 2:9};  % Extract cell array of variable names
            req_vars = req_vars(~cellfun(@isempty, req_vars));  % Remove empty cells
            
            % Load sample data for this land cover type
            sample_file = dir(fullfile(sample_files_path, sprintf('*%s*.csv', lc_name)));
            if isempty(sample_file)
                warning('No sample file found for %s', lc_name);
                continue;
            end
            sample_data = readtable(fullfile(sample_files_path, sample_file(1).name));
            
            % Extract sample NEE and required variables
            sample_nee = sample_data.NEE;
            sample_vars = table2array(sample_data(:, req_vars));
            num_samples = size(sample_vars, 1);
            
            % Get weights (columns 10-17)
            weights = var_selection{pft_row, 10:17};
            weights = weights(~isnan(weights));
            
            % Initialize arrays for global variables
            global_vars = cell(1, length(req_vars));
            pft_vars = cell(1, length(req_vars));
            sdevs = zeros(1, length(req_vars));
            
            % Load each required variable and pre-process
            for v = 1:length(req_vars)
                var_name = req_vars{v};
                
                % Check if it's a soil variable (HWSD2)
                if contains(var_name, 'HWSD2')
                    var_path = fullfile(predictive_vars_path, 'HWSD2', [var_name '.tif']);
                    [var_data, ~] = geotiffread(var_path);
                else
                    % Find monthly file
                    var_folder = fullfile(predictive_vars_path, var_name, '0degree05\');
                    var_files = dir(fullfile(var_folder, sprintf('*%d_%02d*.tif', year, month)));
                    if isempty(var_files)
                        warning('No file found for %s, year %d, month %d', var_name, year, month);
                        continue;
                    end
                    [var_data, ~] = geotiffread(fullfile(var_folder, var_files(1).name));
                end
                
                % Resize if necessary to match landcover dimensions
                if ~isequal(size(var_data), size(landcover))
                    var_data = imresize(var_data, size(landcover), 'nearest');
                end

                % Clean data
                var_data(var_data == -inf | var_data < -3.4028235e+37 | var_data == -32768) = NaN;
                
                if isinteger(var_data)
                    var_data = double(var_data);
                end
                               
                % Calculate standard deviation for similarity weighting
                linear_indices = sub2ind(size(var_data), lc_rows, lc_cols);
                values = var_data(linear_indices);
                sdevs(v) = nanstd(values);
                
                % Store global variables
                pft_vars{v} = double(var_data(linear_indices));
                global_vars{v} = double(var_data);
            end
            
            % Pre-calculate sdevj for each sample and variable
            sdevj = zeros(num_samples, length(req_vars));
            for v = 1:length(req_vars)
                for s = 1:num_samples
                    if isnan(sample_vars(s, v))
                        sdevj(s, v) = NaN;
                    else
                        diff_sq = (pft_vars{v} - sample_vars(s, v)).^2;
                        sdevj(s, v) = sqrt(nansum(diff_sq(:))/sum(~isnan(diff_sq(:))));
                    end
                end
            end
            
            temp_NEE = cell(num_lc_pixels, 1);
            temp_uncertainty = cell(num_lc_pixels, 1);
            temp_positions = zeros(num_lc_pixels, 2);
            
            %
            if isempty(gcp('nocreate'))
                parpool('local', 10); % 10workers
            end
            
            % Process all pixels for this land cover type in parallel
            parfor pix = 1:num_lc_pixels
                row1 = lc_rows(pix);
                col1 = lc_cols(pix);

                pixel_similarities1 = zeros(num_samples, 1);

                % Calculate the similarity with each sample
                for s = 1:num_samples
                    total_sim = 0;
                    valid_vars = 0;
                    
                    for v = 1:length(req_vars)
                        if isempty(global_vars{v}) || isnan(sample_vars(s, v))
                            continue;
                        end
                        
                        pixel_val = global_vars{v}(row1, col1);
                        if isnan(pixel_val)
                            continue;
                        end
                        
                        % Continuous variable similarity
                        diff_sq = (pixel_val - sample_vars(s, v))^2;
                        if sdevj(s, v) == 0
                            var_sim = 0;
                        else
                            var_sim = exp(-diff_sq / (2 * power((sdevs(v) * sdevs(v) / sdevj(s, v)), 2)));
                        end
                        
                        total_sim = total_sim + weights(v) * var_sim;
                        valid_vars = valid_vars + 1;
                    end
                    if valid_vars > 0
                        pixel_similarities1(s) = total_sim;
                    end
                end

                % Select the top 100 most similar samples
                [sorted_sim, sorted_idx] = sort(pixel_similarities1, 'descend');
                valid_sims = sorted_sim(~isnan(sorted_sim) & sorted_sim > 0);
                valid_idx = sorted_idx(~isnan(sorted_sim) & sorted_sim > 0);

                num_top = min(100, length(valid_sims));
                if num_top == 0
                    continue;
                end

                top_sim = valid_sims(1:num_top);
                top_idx = valid_idx(1:num_top);

                temp_positions(pix, :) = [row1, col1];
                temp_uncertainty{pix} = 1 - mean(top_sim);

                % Calculate Weighted NEP
                sum_sim = sum(top_sim);
                if sum_sim > 0
                    norm_weights = top_sim / sum_sim;
                    temp_NEE{pix} = sum(sample_nee(top_idx) .* norm_weights);
                else
                    temp_NEE{pix} = NaN;
                end
            end
            % Merge results into the main matrix
            for pix = 1:num_lc_pixels
                if ~isempty(temp_NEE{pix}) && all(temp_positions(pix, :) > 0)
                    row = temp_positions(pix, 1);
                    col = temp_positions(pix, 2);
                    NEE(row, col) = temp_NEE{pix};
                    NEE_uncertainty(row, col) = temp_uncertainty{pix};
                end
            end
        end
        
        % Save the monthly NEE result
        output_filename = sprintf('%d%02d_NEE.tif', year, month);
        geotiffwrite(fullfile(year_output_path, output_filename), NEE, R);
        
        % Save the monthly NEE_uncertainty result
        output_filename = sprintf('%d%02d_NEE_uncertainty.tif', year, month);
        geotiffwrite(fullfile(year_output_path, output_filename), NEE_uncertainty, R);
    end
end

disp('Processing completed!');