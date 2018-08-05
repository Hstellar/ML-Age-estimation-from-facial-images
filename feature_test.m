clear all
clc

data1=[];
srcFiles = dir('F:\New_Downloads\ML\TestSet\*.jpg');  % the folder in which ur images exists
for i = 1 : length(srcFiles)
    filename = strcat('F:\New_Downloads\ML\TestSet\',srcFiles(i).name);
    
    age=str2double(srcFiles(i).name(5:6));

%Detect objects using Viola-Jones Algorithm
FDetect = vision.CascadeObjectDetector;

EDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',10);
NDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',20);
MDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',160);


I = imread(filename);   %Read the input image
BB = step(FDetect,I);   %To detect Face
if(numel(BB)~=0)    %To Crop Face
    I = imcrop(I,BB(1,:));
end

%To detect Face Features
Ebb = step(EDetect,I);
Nbb = step(NDetect,I);
Mbb = step(MDetect,I);

%Map Different ROI

    Fbb(1)= (Ebb(1)+Nbb(1))/2;      %Map to Forehead reagion
    Fbb(2)= Ebb(2)/3;
    Fbb(3)= Ebb(3)*0.75;
    Fbb(4)= Ebb(4)*0.75;

    MEbb(1)= Fbb(1)+0.4*Fbb(3);     %Map to Mid eyebrow reagion
    MEbb(2)= Fbb(2)+Fbb(4);
    MEbb(3)= Fbb(3)*0.2;
    MEbb(4)= Ebb(4)+Ebb(2)-Fbb(2)-Fbb(4);

    LCbb(1)= Ebb(1);                %Map to Left cheeks reagion
    LCbb(2)= MEbb(2)+MEbb(4);
    LCbb(3)= Fbb(3)*0.5;
    LCbb(4)= Mbb(2)-LCbb(2);

    RCbb(1)= Ebb(1)+Ebb(3)*0.65;    %Map to Right cheeks reagion
    RCbb(2)= LCbb(2);
    RCbb(3)= LCbb(3);
    RCbb(4)= LCbb(4);

    Fbb=abs(Fbb);
    MEbb=abs(MEbb);
    LCbb=abs(LCbb);
    RCbb=abs(RCbb);

    if(size(I,3)==3)                %Convert image to gray scale
        I2=rgb2gray(I);
    else
        I2=I;
    end

    %edge detection for diffrent thresholds
    BW1 = edge(I2,'Canny',0.3);
    BW2 = edge(I2,'Canny',0.2);
    BW3 = edge(I2,'Canny',0.1);
   [BW4,threshOut] = edge(I2,'Canny');
   
    if(size(I,3)==3)    %skin detection for color image
        
        img=rgb2ycbcr(I);
        for i=1:size(img,1)
            for j= 1:size(img,2)
                cb = img(i,j,2);
                cr = img(i,j,3);
                if(~(cr > 132 && cr < 173 && cb > 76 && cb < 126))
                    img(i,j,1)=0;
                    img(i,j,2)=128;
                    img(i,j,3)=128;
                else
                    img(i,j,1)=235;
                    img(i,j,2)=128;
                    img(i,j,3)=128;
                end
            end
        end
        skin=ycbcr2rgb(img);
        skin=logical(rgb2gray(skin));
        
        %Removing edge from non skin area
        BW1=BW1&skin;
        BW2=BW2&skin;
        BW3=BW3&skin;

    end
    
    %find feature value for threshold 1
    FBW=imcrop(BW1,Fbb);
    MEBW=imcrop(BW1,MEbb);
    LCBW=imcrop(BW1,LCbb);
    RCBW=imcrop(BW1,RCbb);
       
    F1=sum(sum(FBW))/numel(FBW)+sum(sum(MEBW))/numel(MEBW)+sum(sum(LCBW))/numel(LCBW)+sum(sum(RCBW))/numel(RCBW);
    
    %find feature value for threshold 2
    FBW=imcrop(BW2,Fbb);
    MEBW=imcrop(BW2,MEbb);
    LCBW=imcrop(BW2,LCbb);
    RCBW=imcrop(BW2,RCbb);
   
    F2=sum(sum(FBW))/numel(FBW)+sum(sum(MEBW))/numel(MEBW)+sum(sum(LCBW))/numel(LCBW)+sum(sum(RCBW))/numel(RCBW);

    %find feature value for threshold 1
    FBW=imcrop(BW3,Fbb);
    MEBW=imcrop(BW3,MEbb);
    LCBW=imcrop(BW3,LCbb);
    RCBW=imcrop(BW3,RCbb);
       
    F3=sum(sum(FBW))/numel(FBW)+sum(sum(MEBW))/numel(MEBW)+sum(sum(LCBW))/numel(LCBW)+sum(sum(RCBW))/numel(RCBW);
    
    %find feature value for threshold 4
    FBW=imcrop(BW4,Fbb);
    MEBW=imcrop(BW4,MEbb);
    LCBW=imcrop(BW4,LCbb);
    RCBW=imcrop(BW4,RCbb);
   
    FF=sum(sum(FBW))/numel(FBW);
    FM=sum(sum(MEBW))/numel(MEBW);
    FL=sum(sum(LCBW))/numel(LCBW);
    FR=sum(sum(RCBW))/numel(RCBW);
    
    F4=sum(sum(FBW))/numel(FBW)+sum(sum(MEBW))/numel(MEBW)+sum(sum(LCBW))/numel(LCBW)+sum(sum(RCBW))/numel(RCBW);
  
    
    %append feature value to data1 matrix
    data1 = vertcat(data1,[age F1 F2 F3 F4 threshOut]);
    

end

%save data1 matrix as .mat file
 m = matfile('F:\New_Downloads\ML\test_res.mat','Writable',true);
 m.data1=data1;