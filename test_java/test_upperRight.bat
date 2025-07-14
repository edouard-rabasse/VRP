@echo off
set NUMBER=5
set SUFFIX=1

set /a NEXT_SUFFIX=%SUFFIX%+1

cd MSH/MSH

java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineUpperRightConstraint ExperimentsAllSets.txt %NUMBER% configurationRectangle.xml


@REM java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt 5 configuration1.xml 1 8

@REM for /L %%i in (1,8,79) do (
@REM     echo Running iteration %%i
@REM     java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt %%i configuration8.xml 1 8
@REM )



cd .. 
cd ..

python src/test/plot_one.py -c "MSH\MSH\instances\Coordinates_%NUMBER%.txt" -a "MSH\MSH\results\configurationUpperRight\Arcs_%NUMBER%_%NEXT_SUFFIX%.txt" 
