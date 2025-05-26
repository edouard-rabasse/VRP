
# This is a script to run the MSH algorithm on Windows using Gurobi

## create instances

    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.CreateInstances 10 10 30 1001 1002

### id not in subfolder

    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "MSH/MSH/bin;C:\gurobi1201\win64\lib\gurobi.jar" main.CreateInstances 10 10 30 1001 1002

# It assumes that the Gurobi library is installed and the path is set correctly

    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt 2 configuration1.xml 1 8

    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.CreateInstances 10 10 50 100 300





    for /L %%i in (1,1,79) do (
    echo %%i 7 
    java -Xmx14000m -Djava.library.path="C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolution_v2 ExperimentsAllSets.txt %%i configuration7.xml 1 8)
    