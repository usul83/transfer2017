load mnist.mat
%imshow(training.images(:,:,108));
[~,~,nTrain] = size(training.images);
[~,~,nTest] = size(test.images);
n = nTrain + nTest;
% concatenate trianing and test sets
allImages = cat(3,training.images,test.images);

% binarize data
allImages(allImages<0.5) = 0;
allImages(allImages~=0) = 1;

% randomly morph each a third of the set thicker and thinner
FatIndices = randperm(n);
ThinIndices = FatIndices(2*int32(n/3)+1:end);
normalIndices = FatIndices(int32(n/3)+1:2*int32(n/3));
FatIndices = FatIndices(1:int32(n/3));

for morphLoop = 1:length(FatIndices)
    allImages(:,:,FatIndices(morphLoop)) = bwmorph...
        (allImages(:,:,FatIndices(morphLoop)),'thicken');
    allImages(:,:,ThinIndices(morphLoop)) = bwmorph...
         (allImages(:,:,ThinIndices(morphLoop)),'thin');
    if rem(morphLoop,1000) == 0
        disp(morphLoop);
    end
end

% randomly add 5% or 10% SD noise to each a third of the set
noiseIndices = randperm(n);
zeroIndices = noiseIndices(2*int32(n/3)+1:end);
tenIndices = noiseIndices(int32(n/3)+1:2*int32(n/3));
twentyIndices = noiseIndices(1:int32(n/3));
noiseTen = rand(28,28,length(tenIndices))*2 -1;
noiseTwenty = noiseTen;
noiseTen(noiseTen<-0.9) = -1;
noiseTen(noiseTen>0.9) = 1;
noiseTen(abs(noiseTen)~=1) = 0;
noiseTwenty(noiseTwenty<-0.8) = -1;
noiseTwenty(noiseTwenty>0.8) = 1;
noiseTwenty(abs(noiseTwenty)~=1) = 0;

allImages(:,:,tenIndices) = allImages(:,:,tenIndices) + noiseTen;
allImages(:,:,twentyIndices) = allImages(:,:,twentyIndices) + noiseTwenty;
allImages(allImages<0.5) = 0;
allImages(allImages~=0) = 1;

imshow(allImages(:,:,tenIndices(1)));
figure
imshow(allImages(:,:,twentyIndices(1)));

%create three contigous one hot vectors for character, thickness and noise
allLabels = [training.labels;test.labels];
newLabels = zeros(n,19);
for OneHotLoop = 1:10
    newLabels(allLabels+1==OneHotLoop,OneHotLoop) = 1;
end

% one-hot vectors
newLabels(ThinIndices,11) = 1;
newLabels(normalIndices,12) = 1;
newLabels(FatIndices,13) = 1;
newLabels(zeroIndices,14) = 1;
newLabels(tenIndices,15) = 1;
newLabels(twentyIndices,16) = 1;

% category labels
newLabels(:,17) = allLabels;
newLabels(ThinIndices,18) = 1;
newLabels(normalIndices,18) = 2;
newLabels(FatIndices,18) = 3;
newLabels(zeroIndices,19) = 1;
newLabels(tenIndices,19) = 2;
newLabels(twentyIndices,19) = 3;


allImages = reshape(allImages,28*28,70000)';
allData = [allImages,newLabels];
% save('mnist2.mat','allImages','allLabels');
%save('mnistTest.mat','allData');

imshow(reshape(allData(zeroIndices(134),1:784),28,28));

% fours = training.images(:,:,training.labels(:,1)==6);
% imshow(fours(:,:,4));

