figure(5);
data = csvread('6E.csv', 1);
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
xlabel('Times Programmed');
xtickformat('10^%d');
xlim([1, 9]);
xticks(1:9);
ylim([1.3, 1.9]);
ylabel('v_{stop}');
zlabel('Test Set Accuracy (%)');
zlim([10, 100]);
title('TiN/Hf(Al)O/Hf/TiN (20nm)'); 
view(45, 45);
