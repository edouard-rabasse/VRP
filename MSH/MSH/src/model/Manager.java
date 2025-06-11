package model;

import java.io.IOException;

/**
 * Class to manage the different algorithms
 */

public class Manager {

	public Manager() throws IOException, InterruptedException {

	}

	/**
	 * Runs the MSH algorithm
	 */

	public Solver_gurobi runMSH_gurobi(String instance_identifier) throws IOException, InterruptedException {

		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH:

		solver.MSH(instance_identifier);

		// Returns the solver instance:

		return solver;
	}

	/**
	 * Try to refine the solution
	 */

	public Solver_gurobi runRefineSolution(String instance_identifier) throws IOException, InterruptedException {

		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH:

		solver.refineSolution(instance_identifier);

		// Returns the solver instance:

		return solver;
	}

	/**
	 * Try to refine the routes
	 */

	public Solver_gurobi runRefineRoutes(String instance_identifier) throws IOException, InterruptedException {

		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH:

		solver.refineRoutes(instance_identifier);

		// Returns the solver instance:

		return solver;
	}

	public Solver_gurobi runRefineWithFixedEdges(String instance_identifier)
			throws IOException, InterruptedException {

		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH:

		solver.refineWithFixedEdges(instance_identifier);

		// Returns the solver instance:

		return solver;
	}

	public Solver_gurobi runWithCustomCosts(String coordinatesFile, String CostFile, String arcsFile, int suffix)
			throws IOException, InterruptedException {
		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH with custom costs:

		solver.runWithCustomCosts(coordinatesFile, CostFile, arcsFile, suffix);

		// Returns the solver instance:

		return solver;
	}
}
