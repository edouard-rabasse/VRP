package model;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import core.RouteAttribute;
import core.Split;
import dataStructures.DataHandler;
import distanceMatrices.ArcModificationMatrix;
import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.MSH;
import split.SplitPLRP;
import split.SplitWithEdgeConstraints;
import util.RouteFromFile;

/**
 * Clean, refactored version of the VRP solver using Gurobi for set
 * partitioning.
 * This class has been restructured to use external helper classes for better
 * separation of concerns.
 * 
 * The main public methods (MSH, refineSolution, refineRoutes,
 * refineWithFixedEdges, runWithCustomCosts)
 * maintain their original signatures and behavior to ensure compatibility.
 * 
 * @author Original: nicolas.cabrera-malik
 * @author Refactored by: Edouard Rabasse
 */
public class Solver_gurobi {

	// Main attributes for backward compatibility
	private String instanceName;
	private String instanceIdentifier;
	private double cpuInitialization;
	private double cpuMshSampling;
	private double cpuMshAssembly;
	private CustomArcCostMatrix customArcCosts;

	/**
	 * Constructor with custom arc costs
	 */
	public Solver_gurobi(CustomArcCostMatrix customArcCosts) {
		this.customArcCosts = customArcCosts;
	}

	/**
	 * Default constructor
	 */
	public Solver_gurobi() {
		this.customArcCosts = new CustomArcCostMatrix();
	}

	/**
	 * Main MSH method - maintains original signature and behavior
	 */
	public void MSH(String instanceIdentif) throws IOException {
		// Store instance information
		storeInstanceInfo(instanceIdentif);

		// Measure initialization time
		double initStartTime = getCurrentTime();
		printMessage("Starting the initialization step...");

		// Initialize VRP context and MSH configuration
		VRPInstanceContext context = new VRPInstanceContext(instanceIdentifier);
		Split split = new SplitPLRP(context.getDistances(), context.getDrivingTimes(),
				context.getWalkingTimes(), context.getData());
		MSHConfiguration mshConfig = new MSHConfiguration(context.getData(), context.getDistances(), split);

		// Complete initialization
		double initEndTime = getCurrentTime();
		cpuInitialization = (initEndTime - initStartTime) / 1000000000;
		printMessage("End of the initialization step...");

		// Execute MSH workflow
		MSHExecutor executor = new MSHExecutor();
		MSHExecutor.MSHExecutionResult result = executor.execute(mshConfig.getMSH());

		cpuMshSampling = result.getSamplingTime();
		cpuMshAssembly = result.getAssemblyTime();

		// Generate reports
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSummary(mshConfig.getMSH(), mshConfig.getAssembler(), context.getData(),
				cpuMshSampling, cpuMshAssembly);
		reporter.printSolution(mshConfig.getMSH(), mshConfig.getAssembler(), context.getData());
	}

	/**
	 * Solution refinement method - maintains original signature and behavior
	 */
	public void refineSolution(String instanceIdentif) throws IOException {
		storeInstanceInfo(instanceIdentif);

		double initStartTime = getCurrentTime();
		printMessage("Starting the initialization step...");

		// Initialize context and configuration for refinement
		VRPInstanceContext context = new VRPInstanceContext(instanceIdentifier);
		Split split = new SplitPLRP(context.getDistances(), context.getDrivingTimes(),
				context.getWalkingTimes(), context.getData());

		String refinerPath = RouteFileUtils.getRefinerFilePath(instanceIdentifier);
		MSHConfiguration mshConfig = new MSHConfiguration(context.getData(), context.getDistances(),
				split, instanceName, refinerPath);

		completeInitializationAndExecute(initStartTime, context, mshConfig);
	}

	/**
	 * Route refinement method - maintains original signature and behavior
	 */
	public void refineRoutes(String instanceIdentif) throws IOException {
		storeInstanceInfo(instanceIdentif);

		double initStartTime = getCurrentTime();
		printMessage("Starting the initialization step...");

		// Initialize context
		VRPInstanceContext context = new VRPInstanceContext(instanceIdentifier);
		Split split = new SplitPLRP(context.getDistances(), context.getDrivingTimes(),
				context.getWalkingTimes(), context.getData());

		// Count routes for refinement
		String routePath = RouteFileUtils.getStandardRouteFilePath(instanceName);
		int numRoutes = RouteFileUtils.countRoutesInFile(routePath);

		MSHConfiguration mshConfig = new MSHConfiguration(context.getData(), context.getDistances(),
				split, instanceName, numRoutes);

		completeInitializationAndExecute(initStartTime, context, mshConfig);
	}

	/**
	 * Fixed edges refinement method - maintains original signature and behavior
	 */
	public void refineWithFixedEdges(String instanceIdentif) throws IOException {
		storeInstanceInfo(instanceIdentif);

		double initStartTime = getCurrentTime();
		printMessage("Starting the initialization step...");

		// Load arc modifications
		ArcModificationMatrix arcModificationMatrix = new ArcModificationMatrix();
		arcModificationMatrix.loadFromFile(GlobalParameters.ARCS_MODIFIED_FOLDER + "Arcs_" +
				instanceName + "_" + GlobalParameters.SEED + ".txt");

		// Initialize context with modifications
		VRPInstanceContext context = new VRPInstanceContext(instanceIdentifier, arcModificationMatrix);
		Split split = new SplitPLRP(context.getDistances(), context.getDrivingTimes(),
				context.getWalkingTimes(), context.getData());
		MSHConfiguration mshConfig = new MSHConfiguration(context.getData(), context.getDistances(), split);

		completeInitializationAndExecute(initStartTime, context, mshConfig);
	}

	/**
	 * Custom costs method - maintains original signature and behavior
	 */
	public void runWithCustomCosts(String instanceIdentifier, String costFile, String arcPath, int suffix)
			throws IOException {
		storeInstanceInfo(instanceIdentifier);

		double initStartTime = getCurrentTime();
		printMessage("Starting the initialization step...");

		// Initialize context and custom cost handler
		VRPInstanceContext context = new VRPInstanceContext(instanceIdentifier, customArcCosts);
		CustomCostHandler costHandler = new CustomCostHandler(context.getData());
		costHandler.loadAndUpdateCosts(costFile, arcPath, context.getDistances());
		costHandler.saveCosts(instanceName, suffix + 1);

		// Initialize split with custom costs
		Split split = new SplitWithEdgeConstraints(context.getDistances(), context.getDrivingTimes(),
				context.getWalkingTimes(), context.getData(),
				costHandler.getCustomCosts());

		// Determine MSH configuration based on arc file availability
		MSHConfiguration mshConfig = createMSHConfigForCustomCosts(context, split, arcPath);

		// Complete initialization
		double initEndTime = getCurrentTime();
		cpuInitialization = (initEndTime - initStartTime) / 1000000000;
		printMessage("End of the initialization step...");

		// Execute MSH
		MSHExecutor executor = new MSHExecutor();
		MSHExecutor.MSHExecutionResult result = executor.execute(mshConfig.getMSH());

		// Calculate original cost and generate analysis
		String initialArcPath = RouteFileUtils.getComparisonFilePath(instanceName);
		Double totalCost = RouteFromFile.getTotalAttribute(RouteAttribute.COST, initialArcPath,
				this.instanceIdentifier);
		System.out.println("Total cost of the solution: " + totalCost);

		// Generate detailed report with cost analysis
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSolutionWithAnalysis(mshConfig.getAssembler(), context.getData(), suffix + 1,
				context.getDistances(), context.getWalkingTimes(),
				this.instanceIdentifier, totalCost);
	}

	// Private helper methods

	private void storeInstanceInfo(String instanceIdentif) {
		this.instanceIdentifier = instanceIdentif;
		this.instanceName = instanceIdentif.replace(".txt", "").replace("Coordinates_", "");
	}

	private double getCurrentTime() {
		return (double) System.nanoTime();
	}

	private void printMessage(String message) {
		if (GlobalParameters.PRINT_IN_CONSOLE) {
			System.out.println(message);
		}
	}

	private void completeInitializationAndExecute(double initStartTime, VRPInstanceContext context,
			MSHConfiguration mshConfig) {
		// Complete initialization timing
		double initEndTime = getCurrentTime();
		cpuInitialization = (initEndTime - initStartTime) / 1000000000;
		printMessage("End of the initialization step...");

		// Execute MSH workflow
		MSHExecutor executor = new MSHExecutor();
		MSHExecutor.MSHExecutionResult result = executor.execute(mshConfig.getMSH());

		cpuMshSampling = result.getSamplingTime();
		cpuMshAssembly = result.getAssemblyTime();

		// Generate reports
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSummary(mshConfig.getMSH(), mshConfig.getAssembler(), context.getData(),
				cpuMshSampling, cpuMshAssembly);
		reporter.printSolution(mshConfig.getMSH(), mshConfig.getAssembler(), context.getData());
	}

	/*
	 * Creates an MSH configuration for custom costs.
	 * This method checks if the arc file exists and is valid.
	 */
	private MSHConfiguration createMSHConfigForCustomCosts(VRPInstanceContext context, Split split, String arcPath) {
		String globalArcPath = RouteFileUtils.getGlobalArcPath(arcPath);

		if (!RouteFileUtils.isRouteFileValid(globalArcPath)) {
			System.out.println("[Solver_gurobi.run] The file with the arcs to be fixed : " + globalArcPath
					+ " does not exist or is not provided. We will run the MSH without fixing arcs.");
			return new MSHConfiguration(context.getData(), context.getDistances(), split);
		} else {
			try {
				int numRoutes = RouteFileUtils.countRoutesInFile(globalArcPath);
				return new MSHConfiguration(context.getData(), context.getDistances(), split, instanceName, numRoutes, 
						globalArcPath);
			} catch (IOException e) {
				System.out.println("Error reading route file, using standard configuration");
				return new MSHConfiguration(context.getData(), context.getDistances(), split);
			}
		}
	}

	// Legacy compatibility methods (for backward compatibility, keep original logic
	// if needed)

	public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data) {
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSolution(msh, assembler, data);
	}

	public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data, int suffix) {
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSolution(msh, assembler, data, suffix);
	}

	public void printSummary(MSH msh, AssemblyFunction assembler, DataHandler data) {
		SolutionReporter reporter = new SolutionReporter(instanceName, cpuInitialization);
		reporter.printSummary(msh, assembler, data, cpuMshSampling, cpuMshAssembly);
	}
}
