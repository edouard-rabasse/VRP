cd MSH/MSH

java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_customCosts Coordinates_5.txt Costs_5_4.txt configurationCustomCosts2.xml Arcs_5_2.txt 4


@REM java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt 5 configuration1.xml 1 8

@REM for /L %%i in (1,8,79) do (
@REM     echo Running iteration %%i
@REM     java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_gurobi ExperimentsAllSets.txt %%i configuration8.xml 1 8
@REM )



cd .. 
cd ..

python src/test/plot_one.py -c "MSH\MSH\instancesCustomCosts\Coordinates_5.txt" -a "MSH\MSH\results\configurationCustomCosts\Arcs_5_2.txt" 