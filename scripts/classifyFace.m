function classifyFace()

addpath('../../../mlab/util/');

workspace_path = '../../../';

[inputData, inputHdr] = readpvpfile(['../outputImgNet/P2.pvp']);

[ffainputData, inputHdr2] = readpvpfile(['../outputImgNet/FFAP2.pvp']);

nneurons = 256;
%nneurons = 8192;
%nneurons = 131072;

Zeronum = zeros(1,nneurons);
allface=[];
allupside = [];

ffaallface=[];
ffaallupside = [];

thresh = 1.4
missed =[];
missedindex=[];
meanratio = [];

for i=1:1000
    X(sub2ind([1,nneurons],inputData{i,1}.values(:,1)+1)) = inputData{i,1}.values(:,2);
    X(numel(Zeronum)) = 0;
    allface = [allface X];
    
    X3(sub2ind([1,nneurons],ffainputData{i,1}.values(:,1)+1)) = ffainputData{i,1}.values(:,2);
    X3(numel(Zeronum)) = 0;
    ffaallface = [ffaallface X3];

   
    disp([i mean(X), mean(X3) mean(X3)/mean(X)])
    ratio = mean(X3)/mean(X);
    meanratio = [meanratio ratio];
    if(ratio >thresh)
        missed = [missed ratio];
        missedindex = [missedindex i];
    end
end

disp(numel(missed));
disp(mean(meanratio));
missed
missedindex
end

