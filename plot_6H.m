figure(8);
data = csvread('6H.csv', 1);
x = log10(data(:, 1));
y = data(:, 2);
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
ylim([5e-10, 5e-8]);
set(gca, 'YScale', 'log')
ylabel('\phi (m)');
zlabel('Test Set Accuracy (%)');
zlim([10, 100]);
zticks(10:30:100);
title('Au/NiO/Si'); 
view(45, 45);