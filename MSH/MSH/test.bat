cd MSH/MSH
java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolutionFixedEdges ExperimentsAllSets.txt 5 config7_2.xml 1 8

java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolution_v2 ExperimentsAllSets.txt 5 configuration7.xml 2 8

cd .. 
cd ..