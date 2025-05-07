@echo off

for /L %%i in (80,1,1000) do (
    echo %%i 1
    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt %%i configuration1.xml 1 8)




for /L %%i in (80,1,1000) do (
    echo %%i 7 
    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolution_v2 ExperimentsAllSets.txt %%i configuration7.xml 1 8)