package model;

import java.io.File;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import java.util.Iterator;

import core.Route;
import core.RouteAttribute;
import core.RoutePool;

import dataStructures.DataHandler;

import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.GurobiSetPartitioningSolver;
import msh.MSH;
import msh.MSHContext;
import util.SolutionPrinter;
import validation.RouteConstraintValidator;
import util.RouteFromFile;
import util.RouteProcessor;
import split.SplitWithEdgeConstraints;
import heuristic.HeuristicConfiguration;
import msh.MSHExecutor;

/**
 * This class contains the main logic of the MSH.
 * It contains the information of the instance and the method MSH.
 * 
 * @author nicolas.cabrera-malik
 */
public class Solver_gurobi {

	// Main attributes
	private String instance_name;
	private String instance_identifier;

	// Statistics
	private double cpu_initialization;
	private double cpu_msh_sampling;
	private double cpu_msh_assembly;

	// Custom cost matrix for Arcs
	// private CustomArcCostMatrix customArcCosts;

	// Constructors
	// public Solver_gurobi(CustomArcCostMatrix customArcCosts) {
	// this.customArcCosts = customArcCosts;
	// }

	public Solver_gurobi() {
		// this.customArcCosts = new CustomArcCostMatrix();
	}

	/**
	 * Common initialization logic shared across all MSH methods
	 */
	private MSHContext initialize(String instance_identifier) throws IOException {
		// Store main attributes
		this.instance_identifier = instance_identifier;
		this.instance_name = instance_identifier.replace(".txt", "").replace("Coordinates_", "");

		Double iniTime = (double) System.nanoTime();
		printMessage("Starting the initialization step...");

		MSHContext context = MSHContext.initializeMSH(instance_identifier);
		Double finTime = (double) System.nanoTime();
		cpu_initialization = (finTime - iniTime) / 1000000000;
		printMessage("End of the initialization step...");

		return context;
	}

	/**
	 * Common execution logic for MSH phases
	 */
	private void executeMSH(MSH msh, AssemblyFunction assembler, DataHandler data) {
		// Sampling phase
		// executeMSHPhases(msh);
		cpu_msh_sampling = MSHExecutor.executeSampling(msh);
		cpu_msh_assembly = MSHExecutor.executeAssembly(msh);

		// Print results
		printSummary(msh, assembler, data);
		printSolution(msh, assembler, data);
	}

	/**
	 * Main MSH method
	 */
	public void MSH(String instance_identifier) throws IOException {
		MSHContext context = initialize(instance_identifier);

		HeuristicConfiguration.addStandardSamplingFunctions(context);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Add insertion-based sampling function
	 */

	/**
	 * Refine solution method
	 */
	public void refineSolution(String instance_identifier) throws IOException {
		MSHContext context = initialize(instance_identifier);

		String path = "./results/configuration1/Arcs_" + this.instance_name + "_" + GlobalParameters.SEED + ".txt";
		HeuristicConfiguration.addRefinerSamplingFunction(context, path, this.instance_name);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Refine routes method
	 */
	public void refineRoutes(String instance_identifier) throws IOException {
		MSHContext context = initialize(instance_identifier);

		String path = "./results/configuration1/Arcs_" + this.instance_name + "_" + GlobalParameters.SEED + ".txt";
		HeuristicConfiguration.addRouteRefinerSamplingFunctions(context, path, this.instance_name);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	public void refineRoutes2(String instance_identifier) throws IOException {
		initialize(instance_identifier);

		String path = "./results/configuration1/Arcs_" + this.instance_name + "_" + GlobalParameters.SEED + ".txt";

		int numRoutes = RouteProcessor.countRoutesInFile(path);

		// Container for all refined routes
		RoutePool combinedPool = new RoutePool();

		// Base data to use for final assembly
		DataHandler baseData = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

		for (int routeId = 0; routeId < numRoutes; routeId++) {

			MSHContext context = MSHContext.initializeMSH(instance_identifier, path, routeId, baseData);

			cpu_msh_sampling += setPools(context, baseData, combinedPool);
		}

		System.out.println(path + " - Combined pool size: " + combinedPool.size());

		MSHContext globalContext = MSHContext.initializeMSH(instance_identifier);

		// Assemblage final avec toutes les routes
		AssemblyFunction assembler = new GurobiSetPartitioningSolver(baseData.getNbCustomers(), true, baseData);
		MSH globalMSH = new MSH(assembler, GlobalParameters.THREADS);

		// executeSampling(globalMSH);
		globalContext.msh = globalMSH;
		globalContext.assembler = assembler;
		globalContext.pools = new ArrayList<RoutePool>(List.of(combinedPool));

		globalContext.msh.setPools(new ArrayList<RoutePool>(List.of(combinedPool)));

		cpu_msh_assembly = MSHExecutor.executeAssembly(globalContext.msh);
		// printSummary(globalMSH, assembler, baseData);
		printSolution(globalMSH, assembler, baseData);

	}

	/**
	 * Run with custom costs method
	 */
	public void runWithCustomCosts(String instance_identifier, String costFile, String arcPath, int suffix)
			throws IOException {
		MSHContext context = initialize(instance_identifier);

		// Setup custom costs
		setupCustomCosts(context, costFile, arcPath, suffix);
		printMessage("[Debug] Custom costs set up with file: " + costFile);

		// Determine sampling strategy based on file existence
		String globalArcPath = GlobalParameters.RESULT_FOLDER + arcPath;
		if (!new File(globalArcPath).isFile()) {
			HeuristicConfiguration.addStandardSamplingFunctions(context);
		} else {
			HeuristicConfiguration.addRouteRefinerSamplingFunctions(context, globalArcPath, this.instance_name);
		}

		context.msh.setPools(context.pools);
		printMessage("Starting MSH with custom costs...");
		executeMSHWithCustomAnalysis(context, suffix);
		executeCostAnalysis(context, suffix + 1);
	}

	/**
	 * Setup custom cost matrix
	 */
	private void setupCustomCosts(MSHContext context, String costFile, String arcPath, int suffix) {
		CustomArcCostMatrix arcCost = loadCustomCostMatrix(costFile, arcPath, context);
		applyCostMatrixToContext(context, arcCost, suffix);
	}

	/**
	 * Load and prepare the custom cost matrix from files
	 */
	private CustomArcCostMatrix loadCustomCostMatrix(String costFile, String arcPath, MSHContext context) {
		int depot = context.data.getNbCustomers() + 1;

		CustomArcCostMatrix arcCost = new CustomArcCostMatrix();
		arcCost.addDepot(depot);

		try {
			// Load custom costs from file
			arcCost.loadFromFile(GlobalParameters.ARCS_MODIFIED_FOLDER + costFile);
			printMessage("[Debug] Loaded custom costs from: " + costFile);
		} catch (IOException e) {
			System.out.println("Error loading custom costs: " + e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}

		try {
			// Load distances from the specified arc path
			arcCost.updateFromFlaggedFile(GlobalParameters.RESULT_FOLDER + arcPath,
					GlobalParameters.CUSTOM_COST_MULTIPLIER, context.distances,
					GlobalParameters.DEFAULT_WALK_COST, context.data);
			printMessage("[Debug] Updated costs from flagged file: " + arcPath);
		} catch (IOException e) {
			System.out.println("Error updating custom costs from flagged file: " + e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}

		return arcCost;
	}

	/**
	 * Apply the cost matrix to the context, handling mapping if necessary
	 */
	private void applyCostMatrixToContext(MSHContext context, CustomArcCostMatrix arcCost, int suffix) {
		if (context.data.getMapping() != null && !context.data.getMapping().isEmpty()) {
			printMessage("[Debug] Applying mapping to convert global → local costs");

			// Convert global → local (true = mapping is global → local)
			CustomArcCostMatrix localArcCost = arcCost.applyMapping(context.data.getMapping(), true);

			// Use local cost matrix for split
			context.split = new SplitWithEdgeConstraints(context.distances, context.drivingTimes,
					context.walkingTimes, context.data, localArcCost);

			printMessage("[Debug] Using local costs with mapping for split");
		} else {
			printMessage("[Debug] Using global costs directly for split");

			// Use global costs directly
			context.split = new SplitWithEdgeConstraints(context.distances, context.drivingTimes,
					context.walkingTimes, context.data, arcCost);

			// Save the global arc costs only if no mapping is present
			String outputFile = GlobalParameters.ARCS_MODIFIED_FOLDER + "Costs_" + instance_name + "_" + (suffix + 1)
					+ ".txt";
			arcCost.saveFile(outputFile);
			printMessage("[Debug] Saved global costs to: " + outputFile);
		}
	}

	/**
	 * Execute MSH with custom cost analysis
	 */
	private void executeMSHWithCustomAnalysis(MSHContext context, int suffix) {
		// Run MSH phases
		cpu_msh_sampling = MSHExecutor.executeSampling(context.msh);
		cpu_msh_assembly = MSHExecutor.executeAssembly(context.msh);
		printMessage("End of the MSH phases...");
	}

	private void executeCostAnalysis(MSHContext context, int suffix) {

		// Get original solution cost for comparison
		String initialArcPath = GlobalParameters.COMPARISON_FOLDER + "Arcs_" + instance_name + "_1.txt";

		String easyPath = "./results/configuration7_easy/Arcs_" + this.instance_name + "_1.txt";
		try {

			printMessage("[Debug] Running cost analysis with validator");
			Double totalCost = RouteFromFile.getTotalAttribute(RouteAttribute.COST, initialArcPath,
					this.instance_identifier);
			printMessage("Total cost of the solution: " + totalCost);

			// Print solution with cost analysis
			RouteConstraintValidator validator = new RouteConstraintValidator(this.instance_identifier,
					GlobalParameters.CONFIG_FOR_CONSTRAINTS_FOLDER);

			if (new File(easyPath).isFile()) {
				Double easyCost = RouteFromFile.getTotalAttribute(RouteAttribute.COST, easyPath,
						this.instance_identifier);
				SolutionPrinter.printSolutionWithCostAnalysis(context.assembler, context.data, instance_name,
						suffix,
						context.distances, context.walkingTimes, validator, totalCost, easyCost);
			} else {
				SolutionPrinter.printSolutionWithCostAnalysis(context.assembler, context.data, instance_name,
						suffix,
						context.distances, context.walkingTimes, validator, totalCost);
			}

		} catch (IOException e) {
			System.out.println("Error reading comparison arc file: " + e.getMessage());
		}
	}

	/**
	 * Run with upper right constraint
	 */
	public void runRefinedWithUpperRightConstraint(String instance_identifier) throws IOException {
		// MSHContext context = initialize(instance_identifier);
		printMessage("Starting initialization with upper right constraint...");

		this.instance_identifier = instance_identifier;

		this.instance_name = instance_identifier.replace(".txt", "").replace("Coordinates_", "");

		// TODO : change this

		String arc_path = "Arcs_" + this.instance_name + "_1.txt";
		int suffix = 1;

		// Container for all refined routes
		RoutePool combinedPool = new RoutePool();

		// Base data to use for final assembly
		DataHandler baseData = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

		// Create constraint matrix

		// Update split algorithm with constraints

		// String originalArcPath = GlobalParameters.ARCS_MODIFIED_FOLDER + "Arcs_" +
		// instance_name + "_1.txt";
		// TODO : change to RESULT_FOLDER
		String globalArcPath = GlobalParameters.ARCS_MODIFIED_FOLDER + arc_path;
		int numRoutes = RouteProcessor.countRoutesInFile(globalArcPath);

		for (int routeId = 0; routeId < numRoutes; routeId++) {

			MSHContext context = MSHContext.initializeMSH(instance_identifier, globalArcPath, routeId, baseData);

			printMessage("[Debug] Processing route " + routeId + " for instance " + instance_identifier);

			CustomArcCostMatrix arcCost = CustomArcCostMatrix.createUpperRightConstraintMatrix(context.data);

			printMessage("[Debug] Applying upper right constraint matrix to context");
			applyCostMatrixToContext(context, arcCost, 1);

			cpu_msh_sampling += setPools(context, baseData, combinedPool);
		}

		MSHContext globalContext = MSHContext.initializeMSH(instance_identifier);

		// Assemblage final avec toutes les routes
		AssemblyFunction assembler = new GurobiSetPartitioningSolver(baseData.getNbCustomers(), true, baseData);
		MSH globalMSH = new MSH(assembler, GlobalParameters.THREADS);

		// executeSampling(globalMSH);
		globalContext.msh = globalMSH;
		globalContext.assembler = assembler;
		globalContext.pools = new ArrayList<RoutePool>(List.of(combinedPool));

		CustomArcCostMatrix constraintMatrix = CustomArcCostMatrix.createUpperRightConstraintMatrix(baseData);
		applyCostMatrixToContext(globalContext, constraintMatrix, suffix);

		globalContext.msh.setPools(new ArrayList<RoutePool>(List.of(combinedPool)));

		cpu_msh_assembly = MSHExecutor.executeAssembly(globalContext.msh);
		// printSummary(globalMSH, assembler, baseData);
		// printSolution(globalMSH, assembler, baseData);

		// TODO:restore
		// executeCostAnalysis(globalContext, suffix - 1);
		printSolution(globalMSH, assembler, baseData);
	}

	/**
	 * This method tries to refine the routes in a solution
	 * 
	 * @throws IOException
	 */
	public void refineRoutesWithMSH(String instance_identifier, String costFile, String arcPath, int suffix)
			throws IOException {
		this.instance_identifier = instance_identifier;
		this.instance_name = instance_identifier.replace(".txt", "").replace("Coordinates_", "");

		String globalArcPath = GlobalParameters.RESULT_FOLDER + arcPath;
		int numRoutes = RouteProcessor.countRoutesInFile(globalArcPath);

		// Container for all refined routes
		RoutePool combinedPool = new RoutePool();

		// Base data to use for final assembly
		DataHandler baseData = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

		for (int routeId = 0; routeId < numRoutes; routeId++) {

			MSHContext context = MSHContext.initializeMSH(instance_identifier, globalArcPath, routeId, baseData);

			setupCustomCosts(context, costFile, arcPath, suffix);
			cpu_msh_sampling += setPools(context, baseData, combinedPool);
		}

		System.out.println(globalArcPath + " - Combined pool size: " + combinedPool.size());

		MSHContext globalContext = MSHContext.initializeMSH(instance_identifier);

		// Assemblage final avec toutes les routes
		AssemblyFunction assembler = new GurobiSetPartitioningSolver(baseData.getNbCustomers(), true, baseData);
		MSH globalMSH = new MSH(assembler, GlobalParameters.THREADS);

		// executeSampling(globalMSH);
		globalContext.msh = globalMSH;
		globalContext.assembler = assembler;
		globalContext.pools = new ArrayList<RoutePool>(List.of(combinedPool));

		globalContext.msh.setPools(new ArrayList<RoutePool>(List.of(combinedPool)));
		setupCustomCosts(globalContext, costFile, arcPath, suffix);

		cpu_msh_assembly = MSHExecutor.executeAssembly(globalContext.msh);
		// printSummary(globalMSH, assembler, baseData);
		// printSolution(globalMSH, assembler, baseData);
		executeCostAnalysis(globalContext, suffix + 1);
	}

	private double setPools(MSHContext context, DataHandler baseData, RoutePool combinedPool) throws IOException {
		// Créer un contexte pour cette route (mini instance TSP)

		HeuristicConfiguration.addStandardSamplingFunctions(context);

		// Sampling sur cette route
		context.msh.setPools(context.pools);
		cpu_msh_sampling = MSHExecutor.executeSampling(context.msh);

		// Copier les routes dans le pool global
		for (RoutePool pool : context.pools) {
			addPools(pool, combinedPool, context, baseData);
		}

		return cpu_msh_sampling;
	}

	private void addPools(RoutePool pool, RoutePool combinedPool, MSHContext context,
			DataHandler baseData) {
		Iterator<Route> iterator = pool.iterator();
		while (iterator.hasNext()) {
			Route r = iterator.next();

			// System.out.println("[Debug] route " + r.toString());

			Route r_copy = RouteProcessor.convertRouteToGlobalRoute(r, context.data, baseData);

			combinedPool.add(r_copy);
		}
	}

	/**
	 * Print message if console printing is enabled
	 */
	private void printMessage(String message) {
		if (GlobalParameters.PRINT_IN_CONSOLE) {
			System.out.println(message);
		}
	}

	// Keep existing print methods unchanged
	public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data) {
		printSolution(msh, assembler, data, GlobalParameters.SEED);
	}

	public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data, int suffix) {
		String pathArcs = GlobalParameters.RESULT_FOLDER + "Arcs_" + instance_name + "_" + suffix + ".txt";
		System.out.println("Saving the solution in: " + pathArcs);

		try (PrintWriter pwArcs = new PrintWriter(new File(pathArcs))) {
			System.out.println("-----------------------------------------------");
			System.out.println("Total cost: " + assembler.objectiveFunction);
			System.out.println("Routes:");

			int counter = 0;
			for (Route r : assembler.solution) {
				RouteProcessor.processRoute(r, pwArcs, counter, data.getNbCustomers());
				printRouteInfo(r, counter);
				counter++;
			}

			System.out.println("-----------------------------------------------");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Error while printing the final solution");
		}
	}

	/**
	 * Print route information to console
	 */
	private void printRouteInfo(Route route, int routeId) {
		System.out.println("\t\t Route " + routeId + ": " +
				route.getAttribute(RouteAttribute.CHAIN) + " - Cost: " +
				route.getAttribute(RouteAttribute.COST) + " - Duration: " +
				route.getAttribute(RouteAttribute.DURATION) + " - Service time: " +
				route.getAttribute(RouteAttribute.SERVICE_TIME));
	}

	public void printSummary(MSH msh, AssemblyFunction assembler, DataHandler data) {
		String path = GlobalParameters.RESULT_FOLDER + "Summary_" + instance_name + "_" + GlobalParameters.SEED
				+ ".txt";

		try (PrintWriter pw = new PrintWriter(new File(path))) {
			// Print to file
			pw.println("Instance;" + instance_name);
			pw.println("Instance_n;" + data.getNbCustomers());
			pw.println("Seed;" + GlobalParameters.SEED);
			pw.println("InitializationTime(s);" + cpu_initialization);
			pw.println("TotalTime(s);" + (cpu_msh_sampling + cpu_msh_assembly));
			pw.println("TotalDistance;" + assembler.objectiveFunction);
			pw.println("Iterations;" + msh.getNumberOfIterations());
			pw.println("SizeOfPool;" + msh.getPoolSize());
			pw.println("SamplingTime(s);" + cpu_msh_sampling);
			pw.println("AssemblyTime(s);" + cpu_msh_assembly);

			// Print to console
			System.out.println("-----------------------------------------------");
			System.out.println("Instance: " + instance_name);
			System.out.println("Instance_n: " + data.getNbCustomers());
			System.out.println("Seed: " + GlobalParameters.SEED);
			System.out.println("InitializationTime(s): " + cpu_initialization);
			System.out.println("TotalTime(s): " + (cpu_msh_sampling + cpu_msh_assembly));
			System.out.println("TotalDistance: " + assembler.objectiveFunction);
			System.out.println("Iterations: " + msh.getNumberOfIterations());
			System.out.println("SizeOfPool: " + msh.getPoolSize());
			System.out.println("SamplingTime(s): " + cpu_msh_sampling);
			System.out.println("AssemblyTime(s): " + cpu_msh_assembly);
		} catch (Exception e) {
			System.out.println("Error printing the summary: " + e.getMessage());
		}
	}

}

//

//
