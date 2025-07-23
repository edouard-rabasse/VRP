@echo off
set NUMBER=1
set SUFFIX=1

cd MSH/MSH

java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolution_v2 ExperimentsAllSets.txt %NUMBER% configuration7.xml


@REM java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt 5 configuration1.xml 1 8

@REM for /L %%i in (1,8,79) do (
@REM     echo Running iteration %%i
@REM     java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt %%i configuration8.xml 1 8
@REM )



cd .. 
cd ..

python src/test/plot_one.py -c "MSH\MSH\instances\Coordinates_%NUMBER%.txt" -a "MSH\MSH\results\configuration7\Arcs_%NUMBER%_%SUFFIX%.txt" 
