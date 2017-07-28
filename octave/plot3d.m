clear all
close all
clc

a = fopen("data.txt","r");
b = fscanf(a, "%f");
c = reshape(b,3,47);
a = fopen("planes.txt","r");
b = fscanf(a, "%f");

scatter3(c(1,:),c(2,:),c(3,:));
hold on
[x y] = meshgrid(-100:10:100); % Generate x and y data
z = -1/ b(3)*( b(1)*x + b(2)*y + b(4) ); % Solve for z data
surf(x,y,z) %Plot the surface
