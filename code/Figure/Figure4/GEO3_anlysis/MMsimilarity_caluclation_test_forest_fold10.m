clc;
clear;
close all;

% File paths
allsampleFile = 'PFT\forest.csv';

% Load data
allsampleData = readtable(allsampleFile);
allsampleVars = table2array(allsampleData(:, [27, 31, 14, 23, 20, 15, 17, 34])); % Extract environmental variables
sampleNEE = table2array(allsampleData(:, 5)); % NEE in 5th column
siteNames = allsampleData{:, 1}; % Site names in first column

% Define weights for each environmental variable
weights = [0.2, 0.05, 0.18, 0.15, 0.17, 0.15, 0.07, 0.03];  % forest

% Get unique site names
uniqueSites = unique(siteNames);
numSites = length(uniqueSites);

% Assign unique index to each site
siteIndices = zeros(size(siteNames));
for i = 1:numSites
    siteIndices(ismember(siteNames, uniqueSites{i})) = i;
end

% Use crossvalind to group site indices
numFolds = 5;
foldIndices = crossvalind('Kfold', numSites, numFolds);

% Initialize storage for all site predictions and true values
allPredictedNEE = [];
allTrueNEE = [];

% Calculate standard deviations
sdevPCA1 = std(allsampleVars(:, 1), 'omitnan');
sdevPCA2 = std(allsampleVars(:, 2), 'omitnan');
sdevPCA3 = std(allsampleVars(:, 3), 'omitnan');
sdevPCA4 = std(allsampleVars(:, 4), 'omitnan');
sdevPCA5 = std(allsampleVars(:, 5), 'omitnan');
sdevPCA6 = std(allsampleVars(:, 6), 'omitnan');
sdevPCA7 = std(allsampleVars(:, 7), 'omitnan');
sdevPCA8 = std(allsampleVars(:, 8), 'omitnan');

% Loop through folds
for fold = 1:numFolds
    % Get test site indices for current fold
    testSiteIndices = find(foldIndices == fold);
    
    % Get test and training sample indices
    testIdx = ismember(siteIndices, testSiteIndices);
    trainIdx = ~testIdx;
    
    % Extract test and training data
    unpredictedVars = allsampleVars(testIdx, :);
    sampleVars = allsampleVars(trainIdx, :);
    unpredictedNEE = sampleNEE(testIdx);
    sampleNEE_fold = sampleNEE(trainIdx);
    
    % --- Third Law of Geography ---
    % Initialize similarity scores
    numUnpredicted = size(unpredictedVars, 1);
    numSamples = size(sampleVars, 1);
    similarityScores = zeros(numUnpredicted, numSamples);
    currentNEE = [];

    % Loop through unpredicted samples
    for i = 1:numUnpredicted
        % Extract environmental variables for current unpredicted sample
        unpredictedSample = unpredictedVars(i, :);
        
        % Loop through sampled samples to calculate similarity
        for j = 1:numSamples
            % Extract environmental variables for current sampled sample
            sampledSample = sampleVars(j, :);
            
            % Calculate similarity for each environmental variable using Gaussian function
            similarityvars = zeros(1, 8);  % For each environmental variable
        
            sdevjPCA1 = sqrt(sum(power(unpredictedVars(:, 1) - sampledSample(1), 2)) / size(unpredictedVars, 1));
            sdevjPCA2 = sqrt(sum(power(unpredictedVars(:, 2) - sampledSample(2), 2)) / size(unpredictedVars, 1));
            sdevjPCA3 = sqrt(sum(power(unpredictedVars(:, 3) - sampledSample(3), 2)) / size(unpredictedVars, 1));
            sdevjPCA4 = sqrt(sum(power(unpredictedVars(:, 4) - sampledSample(4), 2)) / size(unpredictedVars, 1));
            sdevjPCA5 = sqrt(sum(power(unpredictedVars(:, 5) - sampledSample(5), 2)) / size(unpredictedVars, 1));
            sdevjPCA6 = sqrt(sum(power(unpredictedVars(:, 6) - sampledSample(6), 2)) / size(unpredictedVars, 1));
            sdevjPCA7 = sqrt(sum(power(unpredictedVars(:, 7) - sampledSample(7), 2)) / size(unpredictedVars, 1));
            sdevjPCA8 = sqrt(sum(power(unpredictedVars(:, 8) - sampledSample(8), 2)) / size(unpredictedVars, 1));
            
            similarityvars(1) = exp(-power((unpredictedSample(1) - sampledSample(1)), 2) / (2 * power((sdevPCA1 * sdevPCA1 / sdevjPCA1), 2)));
            similarityvars(2) = exp(-power((unpredictedSample(2) - sampledSample(2)), 2) / (2 * power((sdevPCA2 * sdevPCA2 / sdevjPCA2), 2)));
            similarityvars(3) = exp(-power((unpredictedSample(3) - sampledSample(3)), 2) / (2 * power((sdevPCA3 * sdevPCA3 / sdevjPCA3), 2)));
            similarityvars(4) = exp(-power((unpredictedSample(4) - sampledSample(4)), 2) / (2 * power((sdevPCA4 * sdevPCA4 / sdevjPCA4), 2)));
            similarityvars(5) = exp(-power((unpredictedSample(5) - sampledSample(5)), 2) / (2 * power((sdevPCA5 * sdevPCA5 / sdevjPCA5), 2)));
            similarityvars(6) = exp(-power((unpredictedSample(6) - sampledSample(6)), 2) / (2 * power((sdevPCA6 * sdevPCA6 / sdevjPCA6), 2)));
            similarityvars(7) = exp(-power((unpredictedSample(7) - sampledSample(7)), 2) / (2 * power((sdevPCA7 * sdevPCA7 / sdevjPCA7), 2)));
            similarityvars(8) = exp(-power((unpredictedSample(8) - sampledSample(8)), 2) / (2 * power((sdevPCA8 * sdevPCA8 / sdevjPCA8), 2)));
        
            % Calculate overall similarity score (S) by weighting individual similarities
            similarityScores(i, j) = sum(similarityvars .* weights);
        end
    end

    % Filter out similarity scores (S) less than 0.5, and apply normalization only to rows with S > 0.5
    validSimilarityScores = similarityScores > 1;
    originalSimilarityScores = similarityScores;
    % Loop through each row and normalize only where S > 0.5
    for i = 1:numUnpredicted
        if any(validSimilarityScores(i, :)) % If current row has valid similarity scores (>0.5)
            % Normalize only valid similarity scores so each row sums to 1
            similarityScores(i, validSimilarityScores(i, :)) = similarityScores(i, validSimilarityScores(i, :)) / sum(similarityScores(i, validSimilarityScores(i, :)));
        else
            % If no valid similarity scores, select maximum similarity score for this row and normalize
            [sortedSimilarity, sortedIdx] = sort(similarityScores(i, :), 'ascend');  % Changed to ascending order
            totalSamples = length(sortedSimilarity);

            % Divide samples into ten equal parts
            quintileSize = floor(totalSamples/10);
            quintileRanges = zeros(10, 2);
            for q = 1:10
                startIdx = (q-1)*quintileSize + 1;
                endIdx = q*quintileSize;
                if q == 10
                    endIdx = totalSamples; % Last range includes all remaining samples
                end
                quintileRanges(q, :) = [startIdx, endIdx];
            end

            % Calculate NEE for each range
            for q = 1:10
                currentRange = quintileRanges(q, 1):quintileRanges(q, 2);
                currentIdx = sortedIdx(currentRange);
                currentSimilarity = sortedSimilarity(currentRange);

                % Normalize current range similarity scores
                if sum(currentSimilarity) > 0
                    currentSimilarity = currentSimilarity / sum(currentSimilarity);
                else
                    currentSimilarity = ones(size(currentSimilarity)) / length(currentSimilarity);
                end
                
                % Calculate NEE prediction for current range
                currentNEE(i,q) = sum(currentSimilarity .* sampleNEE_fold(currentIdx)');
            end
        end
    end

    % Calculate average evaluation metrics for each range
    for q = 1:10
        % Calculate Third Law of Geography evaluation metrics
        r2_geo(fold,q) = corr(currentNEE(:,q), unpredictedNEE)^2;
        rmse_geo(fold,q) = sqrt(mean((currentNEE(:,q) - unpredictedNEE).^2));
        bias_geo(fold,q) = abs(mean(currentNEE(:,q) - unpredictedNEE));
        p = polyfit(currentNEE(:,q), unpredictedNEE, 1);
        slope_geo(fold,q) = p(1);
        
        currentRange = quintileRanges(q, 1):quintileRanges(q, 2);
        currentIdx = sortedIdx(currentRange);
        sim_geo(fold,q) = mean(sortedSimilarity(currentRange));
    end
end

% Calculate evaluation metrics for Third Law of Geography
r2_geo_mean = mean(r2_geo);
rmse_geo_mean = mean(rmse_geo);
bias_geo_mean = mean(bias_geo);
sim_geo_mean = mean(sim_geo);
slope_geo_mean = mean(slope_geo);

r2_geo_std = std(r2_geo);
rmse_geo_std = std(rmse_geo);
bias_geo_std = std(bias_geo);