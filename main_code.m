clc
clear
close all

%%
% read image
pic = imread('.\pic\5.png');

% info = dicominfo('aaFCR0245lc_20070919.dcm');  
% X = dicomread(info);  
% pic = uint8(X./4);
% pic = imcomplement(pic);


%%
%This code is used to determine the size of the image.
%Size()used to catch the size of image and determine the channel numbers
%if the image is too huge, change the size of it
[one_m,one_n,one_k] = size(pic);
if one_m < 1000 || one_n < 1000
    pic = pic;
else
    pic=imresize(pic,0.5,'nearest');
end
%subplot() is used to create coordinate
figure(1);subplot(2,2,1);imshow(pic);title('Original Mammogram');

[pic_x, pic_y, pic_n] = size(pic);

%pic_n==3 means the image is colorful
if pic_n == 3
    G=rgb2gray(pic);
else
    G = pic;
end


%G is grayscale, I still a grayscale
I = G;
[W,H,N] = size(G);

%Get the pixel area on the left and right sides of an image
%Split the image into two left and right
I_left = I(1:W,1:round(H/2));
I_right = I(1:W,round(H/2:H));

% img_original is the original grayscale
img_original=G;
%The point where the pixel is 0
count_left = length(find(I_left(:)==0));
count_right= length(find(I_right(:)==0));


%These sentences are used to determine the location of breast
%If we need to deal with a large numbers of image, we can use it process
%image in batches
fliped = 0; 
if count_left>count_right 
    fliped = 1;
     %fliplr() Flip the array from left to right
    img_original_fliped = fliplr(img_original);
else
    %  % The breast is on the left, so this image do not need to converse
    img_original_fliped = img_original;
end

% Use double function to prevent abnormalities caused by data overflow in image operations
j = double(img_original_fliped);
[m,n] = size(j);

%build a matrix in a loop to confirm the numbers of pixel point
gp=zeros(1,256);
for i=1:256    
	gp(i)=length(find(j==(i-1)))/(m*n);
end
Z1 = find(gp == max(gp));

%This loop is used to background segmentation
lk=j;
for p=1:m
    for q=1:n
        % once the pixel larger than Zl+50, this part will be white
        if j(p,q) >=Z1+50
            lk(p,q)=255;
        else
            % once the pixel of some area is less than Zl+50, it turn to
            % black
            lk(p,q)=0;
        end
    end
end
k = uint8(lk);
% the sensences can be used to clear tag and extract the boundary 
binaryImage = imfill(k, 'holes');
binaryImage = ExtractNLargestBlobs(binaryImage, 1);
%figure();imshow(binaryImage);


%%
%Create maskimage to calculate breast area
maskedImage = G;
j2 = double(maskedImage);
[m,n] = size(j2);
lk=j2;

gp2=zeros(1,256);
for i=50:256    
	gp2(i)=length(find(lk==(i-1)))/(m*n);
end
Z2 = find(gp2 == max(gp2));

for p=1:m
    for q=1:n
        if j2(p,q) >=Z2+20
            lk(p,q)=255; 
        else
            lk(p,q)=0;
        end
    end
end
k = uint8(lk);

%Use a median filter to filter out this noise  
%make the boundary smooth 
denois = medfilt2(k,[10 4]);


%The area connected extraction operator, the purpose is to extract the largest target area, remove the label, and the pectoral muscle corner interference
x=5; y=5;
I = im2double(denois);
J = regiongrowing(I,x,y,0.2); 
%Create a square structure element with a length of 8*8
B=strel('square',8);
J = imdilate(J,B);
%J = imfill(j,'holes');
%figure(), imshow(J);

% binaryImage is the processed background
% These sentences are used to extract the region of breast
J_i = J+binaryImage;
for ii = 1:W
    for jj = 1:H
        if J_i(ii,jj) == 1
            ROI(ii,jj) = 1;
        else 
            ROI(ii,jj) = 0;
        end
    end
end
%figure(), imshow(ROI);

%Use the mask ROI to segment the breast from the original image
ROI_pic = uint8(ROI).*G;
%figure(2), imshow(ROI_pic);
%% slic
numClusters = 10;
I = ROI_pic;
[L,NumLabels] = superpixels(I,numClusters, 'Method', 'slic', 'NumIterations', 100);
[Adj_list, Adj_matrix] = buildAdjList(L, NumLabels);
s = regionprops(L);
[Centroids] = buildCentroidsMatrix(NumLabels, s);
figure(2);
clusteredimg = slic_draw_contours(L,I);
subplot(1, 2, 1); imshow(I);title('Breast area after threshold segmentaion');
subplot(1, 2, 2); imshow(clusteredimg);title('Breast area after SLIC segmentaion');
hold on; plotGraph(Centroids, Adj_list);
hold off;
%%
% ROI_area = length(find(ROI_pic ~= 0));
%After SLIC segmentation method,getting the Breast area
%ROI_area is the whole area of breast
ROI_area = s(2).Area+s(4).Area;
%The imadjust() function is used to adjust the brightness of grayscale 
g1=imadjust(ROI_pic,[],[],3);
%figure(), imshow(g1);

I = double(g1) / 255;
p = I;
r = 16;
eps = 0.1^2;
q = zeros(size(I));
% guidefilter can be used to protect the boundary of breast
q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
%q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
%q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);

I_enhanced = (I - q) * 5 + q;

%figure();imshow([I, q, I_enhanced], [0, 1]);

%Convert image to 8-bit unsigned integer
less = im2uint8(I_enhanced);
figure(1),subplot(2,2,2);imshow(less);title('Enhanced breast tissue');
less2 = imadjust(less);
figure(1);subplot(2,2,3),imshow(less2);title('Strong contrast breast tissu');


%Add Gaussian white noise with a mean value of m and a variance of var_gauss
%J1=imnoise(less2,'gaussian',0,0.0005);
%less2=im2uint8(filter2(fspecial('average',1),J1)/255); 
%figure(4);imshow(less2);


gp3=zeros(1,256);
for i=20:250    
	gp3(i)=length(find(less2==(i-1)))/(m*n);
end
%figure(2),bar(0:255,gp3);
cankao = find(gp3 == max(gp3));

%%
%Extract the breast tissue
ROI_roi = less2;
%% slic 
numClusters = 10;
I = ROI_roi;
[L,NumLabels] = superpixels(I,numClusters, 'Method', 'slic', 'NumIterations', 100);
[Adj_list, Adj_matrix] = buildAdjList(L, NumLabels);
s = regionprops(L);
[Centroids] = buildCentroidsMatrix(NumLabels, s);
figure(3);
clusteredimg = slic_draw_contours(L,I);
subplot(1, 2, 1); imshow(I);title('Strong contrast breast tissu');
subplot(1, 2, 2); imshow(clusteredimg);title('SLIC segmentation of breast tissue');
hold on; plotGraph(Centroids, Adj_list);
hold off;
for p=1:m
    for q=1:n
         %once the pixel point is larger than 145, then the area turn to
         %white
        if ROI_roi(p,q) >=145
            ROI_roi(p,q)=255; 
        else
            ROI_roi(p,q)=0;
        end
    end
end
figure(1), subplot(2,2,4),imshow(ROI_roi);title('Final segmented breast tissue');


%%
%%Calculate the percentage of breast tissue
% ROI_roi_area = length(find(ROI_roi ~= 0));
ROI_roi_area = s(1).Area+s(2).Area+s(4).Area;
BIRADS = ROI_roi_area/ROI_area;

if BIRADS <= 0.25
    disp('A');
else
    if BIRADS <= 0.50 && BIRADS > 0.25
        disp('B');
    else
        if BIRADS <= 0.75 && BIRADS > 0.50
            disp('C');
        else
            disp('D');
        end
    end
end
