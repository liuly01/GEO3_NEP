clear
clc
close all

% Set folder path
products_dir = 'products'; % Directory containing the six TIFF files

% Get all TIFF files in the products folder
tif_files = dir(fullfile(products_dir, '*.tif'));

% Check if files exist
if isempty(tif_files)
    error('No TIFF files found in the specified directory');
end

% Display found files
fprintf('Found %d TIFF files:\n', length(tif_files));
for i = 1:length(tif_files)
    fprintf('%d. %s\n', i, tif_files(i).name);
end

% Initialize arrays to store results
latitude_sums = zeros(length(tif_files), 1);
longitude_sums = zeros(length(tif_files), 1);
file_names = cell(length(tif_files), 1);

% Process each TIFF file
for i = 1:length(tif_files)
    try
        % Get current file path
        file_path = fullfile(products_dir, tif_files(i).name);
        file_names{i} = tif_files(i).name;
        
        % Read TIFF file and spatial reference
        [data, R] = geotiffread(file_path);
        
        % Display file information
        fprintf('\nProcessing file %d/%d: %s\n', i, length(tif_files), tif_files(i).name);
        fprintf('Data dimensions: %d x %d\n', size(data, 1), size(data, 2));
        fprintf('Latitude limits: [%.4f, %.4f]\n', R.LatitudeLimits(1), R.LatitudeLimits(2));
        fprintf('Longitude limits: [%.4f, %.4f]\n', R.LongitudeLimits(1), R.LongitudeLimits(2));
        
        % Calculate latitude sum (sum along longitude dimension)
        % This sums data along columns (dimension 2) for each latitude
        latitude_sum = sum(data, 2, 'omitnan'); % Sum along longitude dimension 
        x = size(latitude_sum,1);
        latitude_sums(i,1:x) = latitude_sum;
        
        % Calculate longitude sum (sum along latitude dimension)
        % This sums data along rows (dimension 1) for each longitude
        longitude_sum = sum(data, 1, 'omitnan'); % Sum along latitude dimension
        y = size(longitude_sum,2);
        longitude_sums(1:y,i) = longitude_sum;
        
    catch ME
        fprintf('Error processing file %s: %s\n', tif_files(i).name, ME.message);
        latitude_sums(i) = NaN;
        longitude_sums(i) = NaN;
    end
end