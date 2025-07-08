package model;

import java.io.IOException;
import java.util.ArrayList;

import core.Route;
import globalParameters.GlobalParameters;
import globalParameters.GlobalParametersReader;

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

	// public Solver_gurobi runRefineWithFixedEdges(String instance_identifier)
	// throws IOException, InterruptedException {

	// // Creates a solver instance:

	// Solver_gurobi solver = new Solver_gurobi();

	// // Runs the MSH:

	// solver.refineWithFixedEdges(instance_identifier);

	// // Returns the solver instance:

	// return solver;
	// }

	public Solver_gurobi runWithCustomCosts(String coordinatesFile, String CostFile, String arcsFile, int suffix)
			throws IOException, InterruptedException {
		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH with custom costs:

		solver.runWithCustomCosts(coordinatesFile, CostFile, arcsFile, suffix);

		// Returns the solver instance:

		return solver;
	}

	/**
	 * Runs the MSH algorithm with upper right corner constraint
	 * This prohibits starting walking loops from nodes where x > 5 and y > 5
	 */
	public Solver_gurobi runRefinedWithUpperRightConstraint(String instance_identifier)
			throws IOException, InterruptedException {

		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH with upper right constraint:

		solver.runRefinedWithUpperRightConstraint(instance_identifier);

		// Returns the solver instance:

		return solver;
	}

	public void runRefineEasy(String coordinatesFile, String arcsFile, int suffix)
			throws IOException, InterruptedException {
		int instanceNumber = Integer.parseInt(coordinatesFile.replaceAll("[^0-9]", ""));

		String inputArcFile = "results/configuration1/" + arcsFile;
		String outputArcFile = GlobalParameters.RESULT_FOLDER + "Arcs_" + instanceNumber + "_" + suffix + ".txt";
		SplitEasy modifier = new SplitEasy(coordinatesFile);
		modifier.modifyRoutesFromFile(inputArcFile, outputArcFile);

	}

	public Solver_gurobi runRefineMSH(String coordinatesFile, String costFile, String arcsFile, int suffix)
			throws IOException, InterruptedException {
		// Creates a solver instance:

		Solver_gurobi solver = new Solver_gurobi();

		// Runs the MSH:

		solver.refineRoutesWithMSH(coordinatesFile, costFile, arcsFile, suffix);
		System.out.println("Refinement completed for " + coordinatesFile + " with suffix " + suffix);

		// Returns the solver instance:

		return solver;
	}
}
