% step 1: laoding the images
buildingDir = fullfile(toolboxdir('vision'), 'visiondata', 'building');
buildingScene = imageDatastore(buildingDir);

% step 2: displaying the images that need to be stiched
montage(buildingScene.Files)

% step 3: registering the successive image pairs



% this will involve identification and matching of features then estimate the
% geometric transformation that maps 
% and finaly compute the transformation that maps into the panorama image




% step 3 (a): reading the first image and registering successive image pair
I = readimage(buildingScene, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% initialize features for I (1)
grayimage =  rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix. Note that the projective transform is used here because the building images are fairly close to the camera. Had the scene been captured from a further distance an affine transform would suffice. 
numImages = nume1(buildingScene.Files);
tforms(numImages) =  projective2d(eye(3));

%initialize variable to hold image sizes
imageSize = zeros(numImages,2);

% Iterate over remaining image pairs
for n = 2:numImages
    
    % Store points and features for I(n-1)
    pointsPrevious = points;
    featuresPrevious = features;
    
    % Read I(n)
    I = readimage(buildingScene, n);
    
    % Convert image to grayscale
    grayImage = rgb2hray(I);
    
    % Save image size
    imageSize(n,:) = size(grayImage);
    
    % Detect and extract SURF features for I(n)
    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);
    
    % Find correspondences between I(n) and I(n-1)
    indexPairs =  matchfeatures(features, featuresPrevious, 'Unique', true);
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
   
    % estimate the transformation between T(n) and T(n-1)
    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    
    %Compute T(n) * T(n-1) * ... * T(1)
    tforms(n).T = tforms(n).T * tforms(n-1).T
end

% At this point, all the transformations in tforms are relative to the
% first image. This was a convenient way to image registration procedure

% start by using the projective 2d ouputLimits method to find the output
% limits for each transform. The output limits are then used to
% automatically

% compute the output limits for each transform

for i = 1:nume1(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i, 2)], [imageSize(i, 1)]);
end
% Next, compute the average X limits for each transfornms and find the
% image that is in the center. Only the X limits are used here because the
% scene 

avgXLim = mena(xlim, 2);
[~, idx] = sort(avgXLim);
centerIdx = floor(nume1(tforms)+1/2);

centerImageIdx = idx(centerIdx);
% finally apply the center image's inverse transform to all the others

Tinv = invert(tforms(centerImageIdx));

for i = 1:nume1(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

% next step: initialize the panorama
% now, cerate an initial, empty, panorama into which all the images are
% mapped.
% Use the outputLimits method to compute the minimum and maximum output
% limits over all transformations. These value sare used to automatically
% compute the size of the panorama

for i = 1:nume1(tforms)
    [xlim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% width and height of panorama
width = round(xMax - xMin);
height = round(yMax - yMin);

% initialize the "empty" panorama
panorama = zeros([height width 3], 'like', I);

% next step: create the panorama
% use imwarp to map images into the panorama and use vision.AlphaBlender to
% overlay the images together

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaksSource', 'Input port');

% create a 2-d spatial reference object defining the size of the panorama
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% create the panorama
for i = 1:numImages
    I = readimage(buildingScene, i);
    
    % transform I into the panorama
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    
    % generate a binary mask
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    %overlay the warpedImage onto the panorama
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)

