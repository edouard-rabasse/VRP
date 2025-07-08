package msh;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;

import com.gurobi.gurobi.GRBLinExpr;
import com.gurobi.gurobi.GRBModel;
import com.gurobi.gurobi.GRB;
import com.gurobi.gurobi.GRBEnv;
import com.gurobi.gurobi.GRBException;

import core.Route;
import core.RouteAttribute;
import core.RoutePool;
import core.Solution;
import core.VRPSolution;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import distanceMatrices.CustomArcCostMatrix;

/**
 * This class solves the set partitioning model to assembly the final solution.
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class GurobiSetPartitioningSolver extends AssemblyFunction {

	/**
	 * Gurobi environment
	 */
	protected GRBEnv env;

	protected int nRequests;
	protected boolean hasTerminals;
	protected DataHandler data;

	/**
	 * Matrice de coûts personnalisés pour les arcs
	 */
	protected CustomArcCostMatrix customArcCosts;

	/**
	 * Comment added by Edouard, it is a particular assembly function
	 * 
	 * @param nRequests    the number of requests to be served
	 * @param hasTerminals true if the problem has terminals, false otherwise
	 * @param data         the data handler : containing infos about the
	 *                     coordinates.
	 */

	/**
	 * Constructeur avec coûts personnalisés
	 * 
	 * @param nRequests      the number of requests to be served
	 * @param hasTerminals   true if the problem has terminals, false otherwise
	 * @param data           the data handler : containing infos about the
	 * @param customArcCosts the custom arc costs matrix
	 */
	public GurobiSetPartitioningSolver(int nRequests, boolean hasTerminals, DataHandler data,
			CustomArcCostMatrix customArcCosts) {
		this.nRequests = nRequests;
		this.hasTerminals = hasTerminals;
		this.data = data;
		this.customArcCosts = customArcCosts;
	}

	/**
	 * Constructeur par défaut
	 * 
	 * @param nRequests    the number of requests to be served
	 * @param hasTerminals true if the problem has terminals, false otherwise
	 * @param data         the data handler : containing infos about the
	 *                     coordinates.
	 */
	public GurobiSetPartitioningSolver(int nRequests, boolean hasTerminals, DataHandler data) {
		this(nRequests, hasTerminals, data, new CustomArcCostMatrix());
	}

	public Solution assembleSolution(Solution bound, ArrayList<RoutePool> pools) {

		// Procedure to remove duplicates (Routes that are found by multiple pools).

		// Create a pool:

		RoutePool pool_f = new RoutePool();

		// Populate this new pool:

		for (RoutePool pool : pools) {

			Iterator<Route> iterator = pool.iterator();

			while (iterator.hasNext()) {

				Route r = iterator.next();
				pool_f.add(r);

			}

		}

		// Print the pool of paths (if selected by the user)

		if (GlobalParameters.PRINT_POOLS_TO_FILE) {
			String path = "./output/" + "Pool" + ".txt";
			try {
				PrintWriter pw = new PrintWriter(new File(path));
				pw.println("-----------------");
				Iterator<Route> iterator = pool_f.iterator();

				while (iterator.hasNext()) {

					Route r = iterator.next();
					pw.println(r.toString() + " - " + r.getAttribute(RouteAttribute.COST) + " - "
							+ r.getAttribute(RouteAttribute.DURATION));
				}

				pw.close();
			} catch (Exception e) {
				System.out.println("Error printing the pools");
			}

		}

		// Gurobi model:

		try {

			if (GlobalParameters.PRINT_IN_CONSOLE) {
				System.out.println("Building the set partitioning model...");
			}

			// Create the model:

			// Creates the environment:

			env = new GRBEnv();

			// Creates the model:

			GRBModel model = new GRBModel(env);

			// Create covering/partitioning constraints and objective function

			GRBLinExpr[] partitioning_ctr = new GRBLinExpr[nRequests];
			GRBLinExpr objectiveExpr = new GRBLinExpr();

			// Initializes the constraints:

			for (int i = 0; i < nRequests; i++) {
				partitioning_ctr[i] = new GRBLinExpr();
			}

			int start, end;

			// SE routes:

			Iterator<Route> iterator = pool_f.iterator();
			while (iterator.hasNext()) {

				// Recover the route:

				Route r = iterator.next();

				// Creates the variable y:

				model.addVar(0, 1, (double) r.getAttribute(RouteAttribute.COST), GRB.BINARY,
						"y_" + pool_f.getHashCode().compute(r));

			}

			model.update();

			// Add terms to the covering/partitioning constraints and objective function

			iterator = pool_f.iterator();
			while (iterator.hasNext()) {

				// Recover the route:

				Route r = iterator.next();

				// Updates the objective function:

				objectiveExpr.addTerm(((double) r.getAttribute(RouteAttribute.COST)),
						model.getVarByName("y_" + pool_f.getHashCode().compute(r)));
				model.getVarByName("y_" + pool_f.getHashCode().compute(r)).set(GRB.DoubleAttr.Start, 0.0); // We set
																											// them all
																											// to 0
																											// (initially)

				// Capture the route:

				ArrayList<Integer> route = (ArrayList<Integer>) r.getRoute();

				// Update the start and end:

				if (hasTerminals) {
					start = 1;
					end = route.size() - 1;
				} else {
					start = 0;
					end = route.size();
				}

				// Update set partitioning constraints:

				for (int i = start; i < end; i++) {
					partitioning_ctr[route.get(i) - 1].addTerm(1,
							model.getVarByName("y_" + pool_f.getHashCode().compute(r)));
				}

			}

			// Add remaining constraints to the model

			for (int i = 0; i < nRequests; i++) {
				model.addConstr(partitioning_ctr[i], GRB.EQUAL, 1, "ServeCustomer_" + i);
			}

			// Establish the sign:

			model.setObjective(objectiveExpr, GRB.MINIMIZE);

			// Update the model:

			model.update();

			// WARM START:

			VRPSolution initialSolution = (VRPSolution) bound;
			if (initialSolution != null) {
				for (Route route : initialSolution.getRoutes()) {
					int routeCode = pool_f.getHashCode().compute(route);
					model.getVarByName("y_" + routeCode).set(GRB.DoubleAttr.Start, 1.0);
				}
			}

			// Hide the output

			if (GlobalParameters.PRINT_IN_CONSOLE) {
				System.out.println("Printing in the output folder the set partitioning model...");
				model.write("./output/SetPartitioningModel" + ".lp");
			} else {
				model.set(GRB.IntParam.OutputFlag, 0);
			}

			if (GlobalParameters.PRINT_IN_CONSOLE) {
				System.out.println("Finished building the set partitioning model...");
				System.out.println("About to start solving the set partitioning model...");
			}

			// Set time limit and # of threads:

			model.set(GRB.DoubleParam.TimeLimit, GlobalParameters.MSH_ASSEMBLY_TIME_LIMIT);
			model.set(GRB.IntParam.Threads, GlobalParameters.THREADS);

			// Solve model:

			model.optimize();

			if (GlobalParameters.PRINT_IN_CONSOLE) {
				System.out.println("Finished building the set partitioning model...");
			}

			// Store the solution:

			// Objective function value:

			objectiveFunction = model.get(GRB.DoubleAttr.ObjVal);

			// Routes and loads:

			solution = new ArrayList<Route>();

			// Iterate overall the routes:

			iterator = pool_f.iterator();

			while (iterator.hasNext()) {

				Route r = iterator.next();
				if (model.getVarByName("y_" + pool_f.getHashCode().compute(r)).get(GRB.DoubleAttr.X) > 0.5) {
					solution.add(r);
				}
			}

		} catch (GRBException e) {
			e.printStackTrace();
		}

		return null;
	}

}
