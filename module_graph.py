from modulegraph import modulegraph

graph = modulegraph.ModuleGraph()
graph.run_script("train.py")

for node in graph.flatten():
    if node.filename and node.filename.endswith(".py"):
        print(node.filename)
