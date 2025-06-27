@echo off

cd MSH/MSH
for /L %%i in (1,1,79) do (
    java -Xmx2000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineEasy  Coordinates_%%i.txt Arcs_%%i_1.txt configuration7_easy.xml)
cd ..
cd ..