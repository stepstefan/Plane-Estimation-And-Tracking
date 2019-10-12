clear all
close all
clc


a = fopen("planes2.txt","r");
plane = fscanf(a, "%f");%modelparams
a = fopen("data2.txt", "r");
b = fscanf(a, "%f");
points = reshape(b,3,247);%modelpoints
a = fopen("arrow.txt", "r");
c = fscanf(a, "%f");
arrow = reshape(c,3,100);

figure(1)
hold on
[x y] = meshgrid(-100:10:100);
hold on
scatter3(points(1,:),points(2,:),points(3,:));
hold on
%scatter3(arrow(1,:),arrow(2,:),arrow(3,:));
hold on
z = -1/ plane(3)*( plane(1)*x + plane(2)*y + plane(4) );
hold on
surf(x,y,z);