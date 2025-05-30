cd MSH/MSH

java -Xmx14000m "-Djava.library.path=C:\gurobi1201\win64\bin" -cp "bin;C:\gurobi1201\win64\lib\gurobi.jar" main.Main_refineSolutionFixedEdges ExperimentsAllSets.txt 17 configuration8.xml 1 8



cd .. 
cd ..

python src/graph/graph_creator.py +variants=modified