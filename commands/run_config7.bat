@echo off

cd MSH/MSH
for /L %%i in (1,1,79) do (
    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolution_v2 ExperimentsAllSets.txt %%i configuration7.xml 1 8)
cd ..
cd ..