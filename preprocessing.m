clear all
clc

data=[];
srcFiles = dir('F:\New_Downloads\ML\DataSet\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('F:\New_Downloads\ML\DataSet\',srcFiles(i).name);
    
    age=str2double(srcFiles(i).name(5:6));

%Detect objects using Viola-Jones Algorithm
    EDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',10);
    NDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',20);
    MDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',160);


%Read the input image
    I = imread(filename);

%To detect Face
    FDetect = vision.CascadeObjectDetector;
    Face = step(FDetect,I);
    %deleat face with no face detection
    if(size(Face,1)==0)
          delete(filename);
    else      
        %crop face
        imgFace = (I(Face(1,2):Face(1,2)+Face(1,4),Face(1,1):Face(1,1)+Face(1,3),:));

        %To detect Face Features
        Ebb = step(EDetect,imgFace);
        Nbb = step(NDetect,imgFace);
        Mbb = step(MDetect,imgFace);

        %deleat face with no feature detection
        if(size(Ebb,1)==0||size(Nbb,1)==0||size(Mbb,1)==0)
            delete(filename);
        end
    end
end


