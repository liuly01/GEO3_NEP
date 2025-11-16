clear
clc
close all

% Set folder path
input_dir = 'GEO3_NEE';

% Get all year subfolders (2001-2024)
year_folders = dir(fullfile(input_dir, '20*'));
year_folders = year_folders([year_folders.isdir]); % Keep only folders
year_folders = year_folders(~ismember({year_folders.name}, {'.', '..'})); % Exclude current and parent directories

% Sort by year
[~, idx] = sort({year_folders.name});
year_folders = year_folders(idx);

% Initialize result matrix (24 years ¡Á 12 months)
uncertainty_means = zeros(24, 12);

% Process each year folder
for year_idx = 1:24
    current_year = year_folders(year_idx).name;
    input_year_dir = fullfile(input_dir, current_year);
    
    fprintf('Processing year: %s\n', current_year);
    
    % Get all uncertainty files for current year
    uncertainty_files = dir(fullfile(input_year_dir, '*uncertainty.tif'));
    
    if isempty(uncertainty_files)
        fprintf('No uncertainty files found in folder %s\n', input_year_dir);
        continue;
    end
    
    % Process each monthly file
    for i = 1:length(uncertainty_files)
        try
            % Read current file
            file_path = fullfile(input_year_dir, uncertainty_files(i).name);
            uncertainty_data = geotiffread(file_path);
            
            % Calculate mean excluding NaN values
            valid_data = uncertainty_data(~isnan(uncertainty_data));
            
            if ~isempty(valid_data)
                uncertainty_means(year_idx, i) = mean(valid_data);
            else
                uncertainty_means(year_idx, i) = NaN;
                fprintf('All data in file %s are NaN\n', uncertainty_files(i).name);
            end
            
            fprintf('  Month %d: Mean = %.4f\n', i, uncertainty_means(year_idx, i));
            
        catch ME
            fprintf('Error processing file %s: %s\n', uncertainty_files(i).name, ME.message);
            uncertainty_means(year_idx, i) = NaN;
        end
    end
end