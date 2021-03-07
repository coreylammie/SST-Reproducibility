figure(3);
data = csvread('6C.csv', 1);
x = log10(data(:, 1));
y = data(:, 2) - 273;
z = data(:, 3);
dt = delaunayTriangulation(x,y) ;
tri = dt.ConnectivityList ;
xi = dt.Points(:,1) ; yi = dt.Points(:,2) ;
F = scatteredInterpolant(x,y,z);
zi = F(xi,yi);
colormap(redblue);
trisurf(tri,xi,yi,zi) 
xlabel('Time (s)');
xtickformat('10^%d');
xticks(0:9);
xlim([1, 9]);
ylim([200, 300]);
yticks(200:25:300);
ylabel('Tempurature (�C)');
zlabel('Test Set Accuracy (%)');
zlim([10, 100]);
zticks(10:30:100);
title('TiN/HfO_x/TiN'); 
view(45, 45);