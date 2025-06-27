@echo off
set NUMBER=8
set SUFFIX=1

set /a NEXT_SUFFIX=%SUFFIX%+1

cd MSH/MSH


java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineEasy Coordinates_%NUMBER%.txt Arcs_%NUMBER%_%SUFFIX%.txt configuration7_easy.xml 




cd .. 
cd ..

python src/test/plot_one.py -c "MSH\MSH\instances\Coordinates_%NUMBER%.txt" -a "MSH\MSH\results\configuration7_2\Arcs_%NUMBER%_%SUFFIX%.txt" 