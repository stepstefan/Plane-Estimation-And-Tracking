clear all
close all
clc

a = fopen("data.txt","r");
b = fscanf(a, "%f");
c = reshape(b,3,153);

scatter3(c(1,:),c(2,:),c(3,:));
hold on
[x y] = meshgrid(-100:10:100); % Generate x and y data
z = -1/ -7.71075785e-01*( 3.77257794e-01*x -1*y + -6.55673981e-01 ); % Solve for z data
surf(x,y,z) %Plot the surface