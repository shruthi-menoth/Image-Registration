clc;
clear all;
close all; 

%% Read images and convert to gray scale

Ia = imread('gonzalez1.png');
Ib = imread('gonzalez2.png');

% Convert RGB image to grayscale

Iag = single(rgb2gray(Ia));
Ibg = single(rgb2gray(Ib));
[R,C] = size(Iag);

%% Extract SIFT features of Ia and Ib using vl_sift function

[Fa,Da] = vl_sift(Iag); % Each column of F is a feature frame
[Fb,Db] = vl_sift(Ibg); % Each column of D is a descriptor
d = dist(Da',Db); % Distance between Da's column and Db's column
size(d)
[Y,Z] = min(d); % Least Distance
count = 0; % No. of non-overlapped correspondences
ca = zeros(1,2); % feature coordinates of Ia
cb = zeros(1,2); % feature coordinates of Ib

%% Estimate correspondences using Euclidean distance

img = [Ia,Ib];
for m = 1:length(Y)
    indicator = 1; % Indicator to avoid overlapped correspondences
    for n = 1:length(Z)
        if n~=m && Z(n)==Z(m)
            indicator = 0;
            break;
        end
    end
    if indicator && Y(m) < 46 % Threshold for Euclidean distance
        count = count + 1;
        ca(count,:) = round(Fa(1:2,Z(m)));
        cb(count,:) = round(Fb(1:2,m));
    end
end
count

%Homography 
[Fa_x,Fa_y] = size(Fa);
[Fb_x,Fb_y] = size(Fb);
if Fa_y >= Fb_y
    nkey = Fb_y;
else
    nkey = Fa_y;
end
A_1 = zeros(2*nkey,9);
for i=1:nkey
    A_1(2*i-1,:) = [ca(1,1),ca(1,2),1,0,0,0 -ca(1,1)*cb(1,1),-cb(1,1)*cb(1,2),-cb(1,1)];
    A_1(2*i,:) = [0,0,0,ca(1,1),ca(1,2),1 -ca(1,1)*cb(1,2),-cb(1,2)*ca(1,2),-cb(1,2)];
end
    [U1,D1,V1] = svd(A_1);
    h_1 = V1(:,9);
    H_1 = [h_1(1),h_1(2),h_1(3);h_1(4),h_1(5),h_1(6);h_1(7),h_1(8),h_1(9)]
    %H = reshape(h,3,3);
    Normalized_H_1 = H_1./H_1(3,3)

%% Stitch the two images
x1 = size(Ib,2);
y1 = size(Ib,1);
boundary1 = [1,x1,x1,1 ;1,1,y1,y1 ;1,1,1,1] ;
boundary1_ = inv(H_1) * boundary1 ;
boundary1_(1,:) = boundary1_(1,:) ./ boundary1_(3,:) ;
boundary1_(2,:) = boundary1_(2,:) ./ boundary1_(3,:) ;
ur1 = min([1 boundary1_(1,:)]):max([size(Ia,2) boundary1_(1,:)]) ;
vr1 = min([1 boundary1_(2,:)]):max([size(Ia,1) boundary1_(2,:)]) ;
[u1,v1] = meshgrid(ur1,vr1) ;
Ia1_ = vl_imwbackward(im2double(Ia),u1,v1) ;
z1_ = Normalized_H_1(3,1) * u1 + Normalized_H_1(3,2) * v1 + Normalized_H_1(3,3) ;
u1_ = (Normalized_H_1(1,1) * u1 + Normalized_H_1(1,2) * v1 + Normalized_H_1(1,3)) ./ z1_ ;
v1_ = (Normalized_H_1(2,1) * u1 + Normalized_H_1(2,2) * v1 + Normalized_H_1(2,3)) ./ z1_ ;
Ib1_ = vl_imwbackward(im2double(Ib),u1_,v1_) ;
mass1 = ~isnan(Ia1_) + ~isnan(Ib1_) ;
Ia1_(isnan(Ia1_)) = 0 ;
Ib1_(isnan(Ib1_)) = 0 ;
mosaic1 = (Ia1_ + Ib1_) ./ mass1 ;
figure;
imagesc(mosaic1) ; axis image off ;
title('Normalized DLT') ;

%% Stitch the two images
xN = size(Ib,2);
yN = size(Ib,1);
boundaryN = [1,xN,xN,1 ;1,1,yN,yN ;1,1,1,1] ;
boundaryN_ = inv(H_1) * boundaryN ;
boundaryN_ = inv(H_1) * boundaryN ;
boundaryN_(1,:) = boundaryN_(1,:) ./ boundaryN_(3,:) ;
boundaryN_(2,:) = boundaryN_(2,:) ./ boundaryN_(3,:) ;
urN = min([1 boundaryN_(1,:)]):max([size(Ia,2) boundaryN_(1,:)]) ;
vrN = min([1 boundaryN_(2,:)]):max([size(Ia,1) boundaryN_(2,:)]) ;
[uN,vN] = meshgrid(urN,vrN) ;
IaN_ = vl_imwbackward(im2double(Ia),uN,vN) ;
zN_ = H_1(3,1) * uN + H_1(3,2) * vN + H_1(3,3) ;
uN_ = (H_1(1,1) * uN + H_1(1,2) * vN + H_1(1,3)) ./ zN_ ;
vN_ = (H_1(2,1) * uN + H_1(2,2) * vN + H_1(2,3)) ./ zN_ ;
IbN_ = vl_imwbackward(im2double(Ib),uN_,vN_) ;
massN = ~isnan(IaN_) + ~isnan(IbN_) ;
IaN_(isnan(IaN_)) = 0;
IbN_(isnan(IbN_)) = 0;
mosaicN = (IaN_ + IbN_) ./ massN ;
figure;
imagesc(mosaicN) ; axis image off ;
title('Standard DLT') ;

%%Algorithm for RANSAC

noc = 7; % Number of correspondences used to find a homography
N = fix(log(1-.99)/log(1-(1-.1)^noc)); % Number of trials by 10% rule
M = fix((1-(.1))*count); % Minimum size for the inlier set
d_min = 1e100;
for n = 1:N
    lcv = 1; % Loop control variable
    while lcv % To avoid repeated selection
        r = randi(count,noc,1);
        r = sort(r);
        for m = 1:noc-1
            lcv = lcv*(r(m+1)-r(m));
        end
        lcv = ~lcv;
    end
    
%Homography
    A = zeros(2*noc,9);
    for m = 1:noc
        A(2*m-1:2*m,:)=...
            [0,0,0,-[ca(r(m),:),1],cb(r(m),2)*[ca(r(m),:),1];
            [ca(r(m),:),1],0,0,0,-cb(r(m),1)*[ca(r(m),:),1]];
    end
    
    [U,D,V] = svd(A); %singular value decomposition of A
    h = V(:,9);
    H = [h(1),h(2),h(3);h(4),h(5),h(6);h(7),h(8),h(9)];
    %H = reshape(h,3,3);
   
    d2 = zeros(count,1); % d^2(x_measured, x_true)
    for m = 1:count
        x_true = H*[ca(m,:),1]'; % x_true in HC
        temp = x_true/x_true(3);
        x_true = temp(1:2); % x_true in image plane
        d = cb(m,:)-x_true';
        d2(m) = d(1)^2+d(2)^2;
    end
    [Y Z] = sort(d2);
    if sum(Y(1:M)) < d_min
        d_min = sum(Y(1:M));
        inliers = Z(1:M);
        outliers = Z(M+1:end);
    end
end
H
Normalized_H = H./H(3,3)

% Visualize the inliers and outliers

figure; image(img); truesize; hold on;
for m = inliers'
    plot([ca(m,1),C+cb(m,1)],[ca(m,2),cb(m,2)],'-og','linewidth',1);
end
for m = outliers'
    plot([ca(m,1),C+cb(m,1)],[ca(m,2),cb(m,2)],'-or','linewidth',1);
end
plot([C,C],[1,R],'-k'); hold off

%% Stitch the two images

x = size(Ib,2);
y = size(Ib,1);
boundary = [1,x,x,1 ;1,1,y ,y ;1,1,1,1] ;
boundary_ = inv(H) * boundary ;
boundary_(1,:) = boundary_(1,:) ./ boundary_(3,:) ;
boundary_(2,:) = boundary_(2,:) ./ boundary_(3,:) ;
ur = min([1 boundary_(1,:)]):max([size(Ia,2) boundary_(1,:)]) ;
vr = min([1 boundary_(2,:)]):max([size(Ia,1) boundary_(2,:)]) ;
[u,v] = meshgrid(ur,vr) ;
Ia_ = vl_imwbackward(im2double(Ia),u,v) ;
z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
Ib_ = vl_imwbackward(im2double(Ib),u_,v_) ;
mass = ~isnan(Ia_) + ~isnan(Ib_) ;
Ia_(isnan(Ia_)) = 0 ;
Ib_(isnan(Ib_)) = 0 ;
mosaic = (Ia_ + Ib_) ./ mass ;
figure
imagesc(mosaic) ; axis image off ;
title('Normalized DLT + RANSAC') ; 

