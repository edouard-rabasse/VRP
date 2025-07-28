@echo off
set NUMBER=18
set SUFFIX=44

set /a NEXT_SUFFIX=%SUFFIX%+1

cd MSH/MSH


java -Xmx2000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineMSH Coordinates_%NUMBER%.txt Costs_%NUMBER%_%SUFFIX%.txt configurationCustomCosts2.xml Arcs_%NUMBER%_%SUFFIX%.txt %SUFFIX%


cd ..
cd ..

python src/test/plot_one.py -c "MSH\MSH\instances\Coordinates_%NUMBER%.txt" -a "MSH\MSH\results\configurationCustomCosts\Arcs_%NUMBER%_%NEXT_SUFFIX%.txt" 