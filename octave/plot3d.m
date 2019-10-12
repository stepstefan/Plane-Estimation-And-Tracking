clear all
close all
clc

a = fopen("data.txt","r");
b = fscanf(a, "%f");  
c = reshape(b,3,54);%chessboardpoints
a = fopen("planes.txt","r");
b = fscanf(a, "%f");%chessboardparams
a = fopen("planes2.txt","r");
d = fscanf(a, "%f");%modelparams
a = fopen("data2.txt", "r");
e = fscanf(a, "%f");
f = reshape(e,3,247);%modelpoints

normb = sqrt(b(1) * b(1) + b(2) * b(2) + b(3) * b(3));
normd = sqrt(d(1) * d(1) + d(2) *d(2) + d(3) * d(3));

pr = b(1) * d(1) + b(2) * d(2) + b(3) * d(3);
pr = pr / (normb * normd);
pr = acos(pr);
pr = rad2deg(pr);
pr

figure(1)
%scatter3(c(1,:),c(2,:),c(3,:));
hold on
[x y] = meshgrid(-1:0.1:1); % Generate x and y data
z = -1/ b(3)*( b(1)*x + b(2)*y + b(4) ); % Solve for z data
%surf(x,y,z) %Plot the surface
hold on

%figure(2)
scatter3(f(1,:),f(2,:),f(3,:))
hold on
z = -1/d(3) *( d(1)*x + d(2) *y + d(4));
surf(x,y,z);
