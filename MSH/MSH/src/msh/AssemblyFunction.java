package msh;

import java.util.ArrayList;

import core.Route;
import core.RoutePool;
import core.Solution;


public abstract class AssemblyFunction {
	
	protected abstract Solution assembleSolution(Solution bound, ArrayList<RoutePool> pools);
	
	public double objectiveFunction;
	public double objectiveFunctionPreMIP;
	public ArrayList<Route> solution;

}
