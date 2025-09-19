clc;
clear;
close;

% File paths
allsampleFile = 'D:\han05\data\fluxdata\MM\Classification\PFT\dataset_newevi\wetland.csv';

% Load data
allsampleData = readtable(allsampleFile);
allsampleVars = table2array(allsampleData(:, [14, 30, 27, 17, 20, 29, 15, 16]));
sampleNEE = table2array(allsampleData(:, 5));
siteNames = allsampleData{:, 1};

% Define weights for each environmental variable
weights = [0.17, 0.06, 0.15, 0.08, 0.18, 0.15, 0.08, 0.13];  % wetland

uniqueSites = unique(siteNames);
numSites = length(uniqueSites);

siteIndices = zeros(size(siteNames));
for i = 1:numSites
    siteIndices(ismember(siteNames, uniqueSites{i})) = i;
end

numFolds = 5;
foldIndices = crossvalind('Kfold', numSites, numFolds);

allPredictedNEE = [];
allTrueNEE = [];

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
    similarityScores_origin = [];

    testSiteIndices = find(foldIndices == fold);

    testIdx = ismember(siteIndices, testSiteIndices);
    trainIdx = ~testIdx;

    unpredictedVars = allsampleVars(testIdx, :);
    sampleVars = allsampleVars(trainIdx, :);
    unpredictedNEE = sampleNEE(testIdx);
    sampleNEE_fold = sampleNEE(trainIdx);
    
    % --- GEO3 ---
    % Initialize similarity scores
    numUnpredicted = size(unpredictedVars, 1);
    numSamples = size(sampleVars, 1);
    similarityScores = zeros(numUnpredicted, numSamples);

    % Loop through unpredicted samples
    for i = 1:numUnpredicted
        % Extract environmental variables for the current unpredicted sample
        unpredictedSample = unpredictedVars(i, :);
        
        % Loop through sampled samples to calculate similarity
        for j = 1:numSamples
            % Extract environmental variables for the current sampled sample
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
        
            % Calculate the overall similarity score (S) by weighting the individual similarities
            similarityScores(i, j) = sum(similarityvars .* weights);
        end
    end

    % Filter out similarity scores (S) less than 0.5, and apply normalization only to rows with S > 0.5
    validSimilarityScores = similarityScores > 1;
    % Loop through each row and normalize only where S > 0.5
    for i = 1:numUnpredicted
        if any(validSimilarityScores(i, :)) % Similarity thresholds can be set to filter results.
            % Only standardize valid similarity scores so that the sum of each row equals 1
            similarityScores(i, validSimilarityScores(i, :)) = similarityScores(i, validSimilarityScores(i, :)) / sum(similarityScores(i, validSimilarityScores(i, :)));
        else
            % Select the 100 most similar samples
            [sortedSimilarity, sortedIdx] = maxk(similarityScores(i, :), 100);
            
            similarityScores_origin = [similarityScores_origin;sortedSimilarity];
            
            totalSimilarity = sum(sortedSimilarity);
            normalizedSimilarity = sortedSimilarity / totalSimilarity;
            
            similarityScores(i, sortedIdx) = normalizedSimilarity;
            validSimilarityScores(i, sortedIdx) = 1;
            validSimilarityScores(i, ~sortedIdx) = 0;
        end
    end
    invalidSimilarityScores = ~validSimilarityScores;
    similarityScores(invalidSimilarityScores == 1) = 0;
    
    % Compute the final weighted NEE for each unpredicted sample
    finalNEE_geo = sum(similarityScores .* sampleNEE_fold', 2);
    
    r2_geo(fold,1) = corr(finalNEE_geo, unpredictedNEE)^2;
    rmse_geo(fold,1) = sqrt(mean((finalNEE_geo - unpredictedNEE).^2));
    bias_geo(fold,1) = mean(finalNEE_geo - unpredictedNEE);
    sim_geo(fold,1) = mean(similarityScores_origin,'all');
    p = polyfit(finalNEE_geo, unpredictedNEE, 1);
    slope_geo(fold,1) = p(1); 
end

r2_geo_mean = mean(r2_geo);
rmse_geo_mean = mean(rmse_geo);
bias_geo_mean = mean(bias_geo);
slope_geo_mean = mean(slope_geo);
sim_geo_mean = mean(sim_geo);

r2_geo_std = std(r2_geo);
rmse_geo_std = std(rmse_geo);
bias_geo_std = std(bias_geo);
slope_geo_std = std(slope_geo);
sim_geo_std = std(sim_geo);

disp('Evaluation Criteria for the Third Law of Geography£º');
disp(['R2: ', num2str(r2_geo_mean),'¡À', num2str(r2_geo_std)]);
disp(['RMSE: ', num2str(rmse_geo_mean),'¡À', num2str(rmse_geo_std)]);
disp(['Bias: ', num2str(bias_geo_mean),'¡À', num2str(bias_geo_std)]);
disp(['Slope: ', num2str(slope_geo_mean),'¡À', num2str(slope_geo_std)]);