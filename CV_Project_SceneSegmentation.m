%% Scene Segmentation using KITTI database
% For Computer Vision Project - ANU 2018
% By Arslan Khan and Irtza Suhail
% This code is based on the code provided by Matlab2018
% https://au.mathworks.com/help/vision/examples/semantic-segmentation-using-deep-learning.html
% Originally the code is used to train the CamVid dataset 
% We have made various changes to train KITTI dataset and get desired results.
%% Define classes
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];
%% Load trainning data
imgDir = fullfile('kitti_semseg_unizg','train','rgb');
imds = imageDatastore(imgDir);
I = readimage(imds,1);
%% load labels
labelIDs = camvidPixelLabelIDs();
labelDir = fullfile('kitti_semseg_unizg','train','labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
%% Test one image to see we have loaded everything correctly
C = readimage(pxds,1);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)
pixelLabelColorbar(cmap,classes);
%% Visualising the data from KITTI dataset
tbl = countEachLabel(pxds) % Count of labels
frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% Resize all test and label images to 360x480
imageFolder = fullfile('imagesResized',filesep);
imds = resizeKITTIImages(imds,imageFolder);

labelFolder = fullfile('labelsResized',filesep);
pxds = resizeKITTIPixelLabels(pxds,labelFolder);

%% Prepare the training and test sets (60-40 ratio)
[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionKITTIData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%% Check if vgg16 is loaded. Make sure to install VGG module beforehand
vgg16();

%% Create the network
imageSize = [360 480 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize,numClasses,'vgg16');

%% Balance Classes Using Class Weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights)

lgraph = removeLayers(lgraph,'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');

%% Training Options
options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',100, ...  
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',2,...
     'Plots','training-progress');
 
%% Data augmentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

%% Start Training (WARNING! It took ~8 hours on our Nvidia Geforce 960M laptop)
%We will try to load our trained network in the zip file if size permits so
%that you don't have to wait for the training.

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

% [net, info] = trainNetwork(pximds,lgraph,options);
 
%% Test Network on 1 random test Image
I = read(imdsTest);
C = semanticseg(I, net);
labelIDs = camvidPixelLabelIDs();
cmap = camvidColorMap;
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);

%% Compare result with ground truth
% expectedResult = read(pxdsTest);
% actual = uint8(C);
% expected = uint8(expectedResult);
% imshowpair(actual, expected)
% 
% iou = jaccard(C, expectedResult);
% table(classes,iou)
%% Provides overall accuracies (WARNING uses alot of memory)
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics

%%
function labelIDs = camvidPixelLabelIDs()
% Return the label IDs corresponding to each class.
%
% The KITTI dataset has 11 classes.
%
% The 11 classes are:
%   "Sky" "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol",
%   "Fence", "Car", "Pedestrian",  and "Bicyclist".
%
% KITTI pixel label IDs are provided as RGB color values. Group them into
% 11 classes and return them as a cell array of M-by-3 matrices.
labelIDs = { ...
    
    % "Sky"
    [
    128 128 128; ... % "Sky"
    ]
    
    % "Building" 
    [
    
    128 000 000; ... % "Building"
    ]
    
    % "Pole"
    [
    192 192 128; ... % "Column_Pole"
    ]
    
    % Road
    [
    128 064 128; ... % "Road"
    ]
    
    % "Pavement"
    [
    000 000 192; ... % "Sidewalk" 
    ]
        
    % "Tree"
    [
    128 128 000; ... % "Tree"
    ]
    
    % "SignSymbol"
    [
    192 128 128; ... % "SignSymbol"
    ]
    
    % "Fence"
    [
    064 064 128; ... % "Fence"
    ]
    
    % "Car"
    [
    064 000 128; ... % "Car"

    ]
    
    % "Pedestrian"
    [
    064 064 000; ... % "Pedestrian"
    ]
    
    % "Bicyclist"
    [
    000 128 192; ... % "Bicyclist"

    ]
    
    };
end


function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.
colormap(gca,cmap)
% Add colorbar to current figure.
c = colorbar('peer', gca);
% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);
% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;
% Remove tick mark.
c.TickLength = 0;
end
function cmap = camvidColorMap()
% Define the colormap used by KITTI dataset.
cmap = [
    128 128 128   % Sky
    128 0 0       % Building
    192 192 128  % Pole
    128 64 128    % Road
    0 0 192      % Pavement
    128 128 0     % Tree
    192 128 128   % SignSymbol
    64 64 128     % Fence
    64 0 128      % Car
    64 64 0       % Pedestrian
    0 128 192     % Bicyclist
    ];
% Normalize between [0 1].
cmap = cmap ./ 255;
end


function imds = resizeKITTIImages(imds, imageFolder)
% Resize images to [360 480].
if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end
reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    I = imresize(I,[360 480]);    
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end
imds = imageDatastore(imageFolder);
end


function pxds = resizeKITTIPixelLabels(pxds, labelFolder)
% Resize pixel label data to [360 480].
classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
if ~exist(labelFolder,'dir')
    mkdir(labelFolder)
else
    pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
    return; % Skip if images already resized
end
reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    L = imresize(L,[360 480],'nearest');
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(L,[labelFolder filename ext])
end
labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end

function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionKITTIData(imds,pxds)
% Partition KITTI data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);
% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);
% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);
% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);
% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);
% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end
