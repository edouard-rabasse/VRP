package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import core.ArrayDistanceMatrix;
import core.InsertionHeuristic;
import core.NNHeuristic;
import core.OrderFirstSplitSecondHeuristic;
import core.RefinerHeuristic;
import core.RefinerHeuristicRoutes;
import core.Route;
import core.RouteAttribute;
import core.RoutePool;
import core.Split;
import core.TSPSolution;
import core.TSPHeuristic;
import dataStructures.DataHandler;
import dataStructures.DataHandlerHighlighted;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDistanceMatrixV2;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import distanceMatrices.ArcModificationMatrix;
import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.GurobiSetPartitioningSolver;
import msh.MSH;
import msh.OrderFirstSplitSecondSampling;
import split.SplitPLRP;
import util.SolutionPrinter;
import validation.RouteConstraintValidator;
import util.RouteFromFile;
import split.SplitWithEdgeConstraints;

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
	private CustomArcCostMatrix customArcCosts;

	// Configuration for different heuristic types
	private enum HeuristicConfig {
		HIGH_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_HIGH, GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN),
		LOW_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_LOW, GlobalParameters.MSH_RANDOM_FACTOR_LOW);

		private final int randomFactor;
		private final int nnRandomFactor;

		HeuristicConfig(int randomFactor, int nnRandomFactor) {
			this.randomFactor = randomFactor;
			this.nnRandomFactor = nnRandomFactor;
		}
	}

	// Constructors
	public Solver_gurobi(CustomArcCostMatrix customArcCosts) {
		this.customArcCosts = customArcCosts;
	}

	public Solver_gurobi() {
		this.customArcCosts = new CustomArcCostMatrix();
	}

	/**
	 * Common initialization logic shared across all MSH methods
	 */
	private MSHContext initializeMSH(String instance_identifier) throws IOException {
		// Store main attributes
		this.instance_identifier = instance_identifier;
		this.instance_name = instance_identifier.replace(".txt", "").replace("Coordinates_", "");

		Double iniTime = (double) System.nanoTime();
		printMessage("Starting the initialization step...");

		// Read the instance
		DataHandler data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

		// Create distance matrices
		ArrayDistanceMatrix distances = new DepotToCustomersDistanceMatrix(data);
		ArrayDistanceMatrix drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
		ArrayDistanceMatrix walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);

		// Initialize pools and assembler
		ArrayList<RoutePool> pools = new ArrayList<RoutePool>();
		AssemblyFunction assembler = new GurobiSetPartitioningSolver(data.getNbCustomers(), true, data);
		MSH msh = new MSH(assembler, GlobalParameters.THREADS);

		// Initialize split algorithm
		Split split = new SplitPLRP(distances, drivingTimes, walkingTimes, data);

		// Calculate iterations
		int numIterations = Math.max(1, (int) Math.ceil(GlobalParameters.MSH_NUM_ITERATIONS / 8.0));

		Double finTime = (double) System.nanoTime();
		cpu_initialization = (finTime - iniTime) / 1000000000;
		printMessage("End of the initialization step...");

		return new MSHContext(data, distances, drivingTimes, walkingTimes, pools, assembler, msh, split, numIterations);
	}

	/**
	 * Common execution logic for MSH phases
	 */
	private void executeMSH(MSH msh, AssemblyFunction assembler, DataHandler data) {
		// Sampling phase
		Double iniTime = (double) System.nanoTime();
		printMessage("Start of the sampling step...");
		msh.run_sampling();
		printMessage("End of the sampling step...");
		Double finTime = (double) System.nanoTime();
		cpu_msh_sampling = (finTime - iniTime) / 1000000000;

		// Assembly phase
		iniTime = (double) System.nanoTime();
		printMessage("Start of the assembly step...");
		msh.run_assembly();
		printMessage("End of the assembly step...");
		finTime = (double) System.nanoTime();
		cpu_msh_assembly = (finTime - iniTime) / 1000000000;

		// Print results
		printSummary(msh, assembler, data);
		printSolution(msh, assembler, data);
	}

	/**
	 * Main MSH method
	 */
	public void MSH(String instance_identifier) throws IOException {
		MSHContext context = initializeMSH(instance_identifier);

		addStandardSamplingFunctions(context);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Adds standard sampling functions (both high and low randomization)
	 */
	private void addStandardSamplingFunctions(MSHContext context) {
		addSamplingFunctions(context, HeuristicConfig.HIGH_RANDOMIZATION, "high");
		addSamplingFunctions(context, HeuristicConfig.LOW_RANDOMIZATION, "low");
	}

	/**
	 * Generic method to add sampling functions with specified randomization level
	 */
	private void addSamplingFunctions(MSHContext context, HeuristicConfig config, String suffix) {
		String[] heuristicTypes = { "NEAREST_INSERTION", "FARTHEST_INSERTION", "BEST_INSERTION" };
		String[] prefixes = { "rni", "rfi", "rbi" };

		// Add NN heuristic
		addNNSamplingFunction(context, config, suffix);

		// Add insertion heuristics
		for (int i = 0; i < heuristicTypes.length; i++) {
			addInsertionSamplingFunction(context, config, heuristicTypes[i], prefixes[i], suffix);
		}
	}

	/**
	 * Add Nearest Neighbor sampling function
	 */
	private void addNNSamplingFunction(MSHContext context, HeuristicConfig config, String suffix) {
		Random random = new Random(GlobalParameters.SEED + getRandomSeed("nn", suffix));

		NNHeuristic nn = new NNHeuristic(context.distances);
		nn.setRandomized(true);
		nn.setRandomGen(random);
		nn.setRandomizationFactor(config.nnRandomFactor);
		nn.setInitNode(0);

		addSamplingFunction(context, nn, "rnn_" + suffix);
	}

	/**
	 * Add insertion-based sampling function
	 */
	private void addInsertionSamplingFunction(MSHContext context, HeuristicConfig config,
			String insertionType, String prefix, String suffix) {
		Random random = new Random(GlobalParameters.SEED + getRandomSeed(prefix, suffix));

		InsertionHeuristic heuristic = new InsertionHeuristic(context.distances, insertionType);
		heuristic.setRandomized(true);
		heuristic.setRandomGen(random);
		heuristic.setRandomizationFactor(config.randomFactor);
		heuristic.setInitNode(0);

		addSamplingFunction(context, heuristic, prefix + "_" + suffix);
	}

	/**
	 * Helper method to add a sampling function with its pool
	 */
	private void addSamplingFunction(MSHContext context, TSPHeuristic tspHeuristic, String name) {
		OrderFirstSplitSecondHeuristic heuristic = new OrderFirstSplitSecondHeuristic(tspHeuristic, context.split);
		OrderFirstSplitSecondSampling sampling = new OrderFirstSplitSecondSampling(heuristic, context.numIterations,
				name);

		RoutePool pool = new RoutePool();
		context.pools.add(pool);
		sampling.setRoutePool(pool);
		context.msh.addSamplingFunction(sampling);
	}

	/**
	 * Get random seed based on heuristic type and suffix
	 */
	private int getRandomSeed(String type, String suffix) {
		int baseOffset = suffix.equals("high") ? 90 : 130;
		int typeOffset = switch (type) {
			case "nn" -> 0;
			case "rni" -> 10;
			case "rfi" -> 20;
			case "rbi" -> 30;
			default -> 0;
		};
		return baseOffset + typeOffset + 1000;
	}

	/**
	 * Refine solution method
	 */
	public void refineSolution(String instance_identifier) throws IOException {
		MSHContext context = initializeMSH(instance_identifier);

		String path = "./results/configuration1/Arcs_" + this.instance_name + "_" + GlobalParameters.SEED + ".txt";
		addRefinerSamplingFunction(context, path);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Refine routes method
	 */
	public void refineRoutes(String instance_identifier) throws IOException {
		MSHContext context = initializeMSH(instance_identifier);

		String path = "./results/configuration1/Arcs_" + this.instance_name + "_" + GlobalParameters.SEED + ".txt";
		addRouteRefinerSamplingFunctions(context, path);
		context.msh.setPools(context.pools);

		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Add refiner sampling function
	 */
	private void addRefinerSamplingFunction(MSHContext context, String path) {
		Random random = new Random(GlobalParameters.SEED + 90 + 1000);

		RefinerHeuristic refiner = new RefinerHeuristic(context.distances, this.instance_name, path);
		refiner.setRandomized(true);
		refiner.setRandomGen(random);
		refiner.setRandomizationFactor(GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
		refiner.setInitNode(0);

		addSamplingFunction(context, refiner, "rnn_high");
	}

	/**
	 * Add route refiner sampling functions
	 * The sampling functions
	 */
	private void addRouteRefinerSamplingFunctions(MSHContext context, String path) {
		int numRoutes = countRoutesInFile(path);

		for (int i = 0; i < numRoutes; i++) {
			Random random = new Random(GlobalParameters.SEED + 90 + 1000);

			RefinerHeuristicRoutes refiner = new RefinerHeuristicRoutes(context.distances, this.instance_name, i, path);
			refiner.setRandomized(true);
			refiner.setRandomGen(random);
			refiner.setRandomizationFactor(GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
			refiner.setInitNode(0);

			addSamplingFunction(context, refiner, "rnn_high_" + i);
		}
	}

	/**
	 * Count routes in solution file
	 */
	private int countRoutesInFile(String path) {
		int numRoutes = 0;
		try (BufferedReader buff = new BufferedReader(new FileReader(path))) {
			String line;
			int previousRoute = -1;

			while ((line = buff.readLine()) != null) {
				String[] parts = line.split(";");
				int currentRoute = Integer.parseInt(parts[3]);
				if (previousRoute != currentRoute) {
					numRoutes++;
					previousRoute = currentRoute;
				}
			}
		} catch (IOException e) {
			System.out.println("Error reading solution file: " + e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}
		return numRoutes;
	}

	/**
	 * Run with custom costs method
	 */
	public void runWithCustomCosts(String instance_identifier, String costFile, String arcPath, int suffix)
			throws IOException {
		MSHContext context = initializeMSH(instance_identifier);

		// Setup custom costs
		setupCustomCosts(context, costFile, arcPath, suffix);
		printMessage("[Debug] Custom costs set up with file: " + costFile);

		// Determine sampling strategy based on file existence
		String globalArcPath = GlobalParameters.RESULT_FOLDER + arcPath;
		if (!new File(globalArcPath).isFile()) {
			addStandardSamplingFunctions(context);
		} else {
			addRouteRefinerSamplingFunctions(context, globalArcPath);
		}

		context.msh.setPools(context.pools);
		printMessage("Starting MSH with custom costs...");
		executeMSHWithCustomAnalysis(context, suffix);
	}

	/**
	 * Setup custom cost matrix
	 */
	private void setupCustomCosts(MSHContext context, String costFile, String arcPath, int suffix) {
		int depot = context.data.getNbCustomers() + 1;

		CustomArcCostMatrix arcCost = new CustomArcCostMatrix();
		arcCost.addDepot(depot);
		try {
			// Load custom costs from file
			arcCost.loadFromFile(GlobalParameters.ARCS_MODIFIED_FOLDER + costFile);
		} catch (IOException e) {
			System.out.println("Error loading custom costs: " + e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}

		try {
			// Load distances from the specified arc path
			arcCost.updateFromFlaggedFile(GlobalParameters.RESULT_FOLDER + arcPath,
					GlobalParameters.CUSTOM_COST_MULTIPLIER, context.distances,
					GlobalParameters.DEFAULT_WALK_COST);
		} catch (IOException e) {
			System.out.println("Error updating custom costs from flagged file: " + e.getMessage());
			e.printStackTrace();
			System.exit(0);
		}

		arcCost.saveFile(
				GlobalParameters.ARCS_MODIFIED_FOLDER + "Costs_" + instance_name + "_" + (suffix + 1) + ".txt");

		// Update split algorithm with custom costs
		context.split = new SplitWithEdgeConstraints(context.distances, context.drivingTimes,
				context.walkingTimes, context.data, arcCost);
	}

	/**
	 * Execute MSH with custom cost analysis
	 */
	private void executeMSHWithCustomAnalysis(MSHContext context, int suffix) {
		// Run MSH phases
		executeMSHPhases(context.msh);
		printMessage("End of the MSH phases...");

		// Get original solution cost for comparison
		String initialArcPath = GlobalParameters.COMPARISON_FOLDER + "Arcs_" + instance_name + "_1.txt";

		String easyPath = "./results/configuration7_easy/Arcs_" + this.instance_name + "_1.txt";
		try {
			Double totalCost = RouteFromFile.getTotalAttribute(RouteAttribute.COST, initialArcPath,
					this.instance_identifier);
			System.out.println("Total cost of the solution: " + totalCost);

			// Print solution with cost analysis
			RouteConstraintValidator validator = new RouteConstraintValidator(this.instance_identifier,
					"./config/configuration7.xml");

			if (new File(easyPath).isFile()) {
				Double easyCost = RouteFromFile.getTotalAttribute(RouteAttribute.COST, easyPath,
						this.instance_identifier);
				SolutionPrinter.printSolutionWithCostAnalysis(context.assembler, context.data, instance_name,
						suffix + 1,
						context.distances, context.walkingTimes, validator, totalCost, easyCost);
			} else {
				SolutionPrinter.printSolutionWithCostAnalysis(context.assembler, context.data, instance_name,
						suffix + 1,
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
		MSHContext context = initializeMSH(instance_identifier);
		printMessage("Starting initialization with upper right constraint...");

		// Create constraint matrix
		CustomArcCostMatrix constraintMatrix = createUpperRightConstraintMatrix(context.data);

		// Update split algorithm with constraints
		context.split = new SplitWithEdgeConstraints(context.distances, context.drivingTimes,
				context.walkingTimes, context.data, constraintMatrix);

		String originalArcPath = GlobalParameters.ARCS_MODIFIED_FOLDER + "Arcs_" + instance_name + "_1.txt";
		if (!new File(originalArcPath).isFile()) {
			printMessage("Arc file not found. Running standard MSH without fixing arcs.");
			addStandardSamplingFunctions(context);
		} else {
			printMessage("Arc file found. Adding route refiner sampling functions.");
			addRouteRefinerSamplingFunctions(context, originalArcPath);
		}

		context.msh.setPools(context.pools);
		executeMSH(context.msh, context.assembler, context.data);
	}

	/**
	 * Create constraint matrix for upper right corner
	 */
	private CustomArcCostMatrix createUpperRightConstraintMatrix(DataHandler data) {
		CustomArcCostMatrix constraintMatrix = new CustomArcCostMatrix();
		int depot = data.getNbCustomers() + 1;
		constraintMatrix.addDepot(depot);

		for (int i = 0; i < data.getNbCustomers(); i++) {
			double xCoord = data.getX_coors().get(i);
			double yCoord = data.getY_coors().get(i);

			if (xCoord > 5.0 && yCoord > 5.0) {
				for (int j = 0; j < data.getNbCustomers(); j++) {
					if (i != j) {
						constraintMatrix.addCustomCost(i, j, 2, GlobalParameters.FIXED_ARCS_DISTANCE * 1000);
					}
				}
				printMessage("Applied upper right constraint to node " + i + " at coordinates (" + xCoord + ", "
						+ yCoord + ")");
			}
		}

		return constraintMatrix;
	}

	/**
	 * Execute only MSH phases (sampling and assembly)
	 */
	private void executeMSHPhases(MSH msh) {
		// Sampling phase
		Double iniTime = (double) System.nanoTime();
		printMessage("Start of the sampling step...");
		msh.run_sampling();
		printMessage("End of the sampling step...");
		Double finTime = (double) System.nanoTime();
		cpu_msh_sampling = (finTime - iniTime) / 1000000000;

		// Assembly phase
		iniTime = (double) System.nanoTime();
		printMessage("Start of the assembly step...");
		msh.run_assembly();
		printMessage("End of the assembly step...");
		finTime = (double) System.nanoTime();
		cpu_msh_assembly = (finTime - iniTime) / 1000000000;
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
				processRoute(r, pwArcs, counter, data.getNbCustomers());
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
	 * Process individual route for solution output
	 */
	private void processRoute(Route route, PrintWriter writer, int routeId, int nbCustomers) {
		String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
		chain = chain.replace(" 0 ", " " + (nbCustomers + 1) + " ");
		chain = chain.replace("CD", "" + (nbCustomers + 1));
		chain = chain.replace(" ", "");
		String[] parts = chain.split("[||]");

		for (String part : parts) {
			if (part.length() > 0) {
				processRoutePart(part, writer, routeId);
			}
		}
	}

	/**
	 * Process individual part of route chain
	 */
	private void processRoutePart(String part, PrintWriter writer, int routeId) {
		if (part.contains("->") && !part.contains("---")) {
			// Only driving
			processSimpleMode(part, "->", 1, writer, routeId);
		} else if (!part.contains("->") && part.contains("---")) {
			// Only walking
			processSimpleMode(part, "---", 2, writer, routeId);
		} else if (part.contains("->") && part.contains("---")) {
			// Mixed mode
			processMixedMode(part, writer, routeId);
		}
	}

	/**
	 * Process route part with single mode (driving or walking)
	 */
	private void processSimpleMode(String part, String delimiter, int mode, PrintWriter writer, int routeId) {
		part = part.replace(delimiter, ";");
		String[] arcParts = part.split("[;]");
		for (int arc = 0; arc < arcParts.length - 1; arc++) {
			if (!arcParts[arc].equals(arcParts[arc + 1])) {
				writer.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + mode + ";" + routeId);
			}
		}
	}

	/**
	 * Process route part with mixed modes (driving and walking)
	 */
	private void processMixedMode(String part, PrintWriter writer, int routeId) {
		// Implementation of mixed mode processing (keeping original complex logic)
		part = part.replace("---", ";").replace("->", ":");

		int posInString = 0;
		String tail = "";
		String head = "";
		int mode = -1;
		int lastPos = -1;

		while (posInString < part.length()) {
			char currentChar = part.charAt(posInString);

			if (mode == -1) {
				if (currentChar == ':') {
					mode = 1;
					lastPos = posInString;
				} else if (currentChar == ';') {
					mode = 2;
					lastPos = posInString;
				}
			} else {
				if (currentChar == ':' || currentChar == ';') {
					if (!tail.equals(head)) {
						writer.println(tail + ";" + head + ";" + mode + ";" + routeId);
					}
					posInString = lastPos;
					mode = -1;
					tail = "";
					head = "";
				}
			}

			if (mode == -1 && currentChar != ':' && currentChar != ';') {
				tail += currentChar;
			} else if (currentChar != ':' && currentChar != ';') {
				head += currentChar;
			}

			posInString++;
		}

		if (!tail.equals(head)) {
			writer.println(tail + ";" + head + ";" + mode + ";" + routeId);
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

	/**
	 * Context class to hold MSH-related objects and reduce parameter passing
	 */
	private static class MSHContext {
		final DataHandler data;
		final ArrayDistanceMatrix distances;
		final ArrayDistanceMatrix drivingTimes;
		final ArrayDistanceMatrix walkingTimes;
		final ArrayList<RoutePool> pools;
		final AssemblyFunction assembler;
		final MSH msh;
		Split split; // Not final as it may be replaced with custom implementations
		final int numIterations;

		MSHContext(DataHandler data, ArrayDistanceMatrix distances, ArrayDistanceMatrix drivingTimes,
				ArrayDistanceMatrix walkingTimes, ArrayList<RoutePool> pools, AssemblyFunction assembler,
				MSH msh, Split split, int numIterations) {
			this.data = data;
			this.distances = distances;
			this.drivingTimes = drivingTimes;
			this.walkingTimes = walkingTimes;
			this.pools = pools;
			this.assembler = assembler;
			this.msh = msh;
			this.split = split;
			this.numIterations = numIterations;
		}
	}
}

//

//
