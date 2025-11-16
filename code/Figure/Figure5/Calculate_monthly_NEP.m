clear
clc
close all

% Set input and output folder paths
input_dir = 'GEO3_NEE'; % Root directory containing year subfolders
output_root = 'GEO3_NEE'; % Root directory for output files

% Start parallel pool with 10 workers
if isempty(gcp('nocreate'))
    parpool(10);
end

% Get all year subfolders (2001-2024)
year_folders = dir(fullfile(input_dir, '20*'));
year_folders = year_folders([year_folders.isdir]); % Keep only folders
year_folders = year_folders(~ismember({year_folders.name}, {'.', '..'})); % Exclude current and parent directories

% Sort by year
[~, idx] = sort({year_folders.name});
year_folders = year_folders(idx);

% Pre-calculate area grid (only need to calculate once)
% Use first file of first year to get spatial reference information
first_year_dir = fullfile(input_dir, year_folders(1).name);
first_NEE_files = dir(fullfile(first_year_dir, [year_folders(1).name '*_NEE.tif']));

if isempty(first_NEE_files)
    error('No NEE files found in first year folder, cannot get spatial reference information');
end

first_file = fullfile(first_year_dir, first_NEE_files(1).name);
[~, R] = geotiffread(first_file);

% Calculate area grid
area_grid = zeros(R.RasterSize);
LAT = R.LatitudeLimits(1, 2); % Latitude of northernmost grid cell
LON = R.LongitudeLimits(1, 1); % Longitude of westernmost grid cell

% Calculate area of each grid cell (m?)
for j = 1:size(area_grid, 1)
    for k = 1:size(area_grid, 2)
        lat1 = LAT - (j-1) * R.CellExtentInLatitude;
        lon1 = LON + (k-1) * R.CellExtentInLongitude;
        lat2 = LAT - j * R.CellExtentInLatitude;
        lon2 = LON + k * R.CellExtentInLongitude;
        
        % Use spherical area formula to calculate grid area
        area_grid(j, k) = (pi/180) * 6371010.162^2 * ...
                         abs(sind(lat2) - sind(lat1)) * ...
                         abs(lon2 - lon1);
    end
end

fprintf('Area grid calculation completed, size: %d x %d\n', size(area_grid, 1), size(area_grid, 2));

% Initialize current year NEP sum
monthly_NEP = zeros(24, 12);

% Process each year folder
for year_idx = 1:24
    current_year = year_folders(year_idx).name;
    input_year_dir = fullfile(input_dir, current_year);
    output_year_dir = fullfile(output_root, current_year, 'NEP\');
    
    % Ensure output folder exists
    if ~exist(output_year_dir, 'dir')
        mkdir(output_year_dir);
    end
    
    % Get all NEE files for current year
    NEE_files = dir(fullfile(input_year_dir, [current_year '*_NEP.tif']));
    
    if isempty(NEE_files)
        fprintf('No NEE files found in folder %s, skipping processing\n', input_year_dir);
        continue;
    end
    
    % Use parfor to process all files for current year in parallel
    parfor i = 1:12
        try
            % Read current file
            file_path = fullfile(input_year_dir, NEE_files(i).name);
            NEE_data = geotiffread(file_path);
            
            % Convert NEE data from gC m-2 month-1 to gC month-1
            converted_NEE = -NEE_data .* area_grid;
            monthly_NEP(year_idx,i) = sum(sum(converted_NEE, 'omitnan'));
            
            % Construct output filename
            [~, name, ext] = fileparts(NEE_files(i).name);
            output_name = strrep(name, '_NEE', '_NEP');
            output_path = fullfile(output_year_dir, [output_name ext]);
            
            % Save converted data as GeoTIFF
            geotiffwrite(output_path, converted_NEE, R);
            
            % Display progress
            fprintf('Year %s: Processed file: %s\n', current_year, NEE_files(i).name);
        catch ME
            fprintf('Error processing file %s: %s\n', NEE_files(i).name, ME.message);
            monthly_NEP(year_idx, i) = NaN;
        end
    end
    
    % Display processing results for current year
    fprintf('Year %s processing completed! NEP values: %s\n', current_year, mat2str(monthly_NEP));
end

% Close parallel pool
delete(gcp('nocreate'));

disp('All year files processing completed!');