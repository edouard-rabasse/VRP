package model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import core.ArrayDistanceMatrix;
import core.InsertionHeuristic;
import core.NNHeuristic;
import core.OrderFirstSplitSecondHeuristic;
import core.RefinerHeuristic;
import core.RefinerHeuristicRoutes;
import core.RoutePool;
import core.Split;
import dataStructures.DataHandler;
import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.GurobiSetPartitioningSolver;
import msh.MSH;
import msh.OrderFirstSplitSecondSampling;

/**
 * Handles the initialization and configuration of MSH (Multi-Start Heuristic)
 * components.
 * This class encapsulates the creation of sampling functions, route pools, and
 * the MSH object itself.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class MSHConfiguration {

    private final MSH msh;
    private final ArrayList<RoutePool> pools;
    private final AssemblyFunction assembler;

    /**
     * Creates a standard MSH configuration with high and low randomization sampling
     * functions
     */
    public MSHConfiguration(DataHandler data, ArrayDistanceMatrix distances, Split split) {
        this.pools = new ArrayList<RoutePool>();
        this.assembler = new GurobiSetPartitioningSolver(data.getNbCustomers(), true, data);
        this.msh = new MSH(assembler, GlobalParameters.THREADS);

        int numIterations = calculateNumIterations();

        // Add sampling functions with different randomization levels
        addHighRandomizationSamplingFunctions(data, distances, split, numIterations);
        addLowRandomizationSamplingFunctions(data, distances, split, numIterations);

        msh.setPools(pools);
    }

    /**
     * Creates an MSH configuration for solution refinement
     */
    public MSHConfiguration(DataHandler data, ArrayDistanceMatrix distances, Split split,
            String instanceName, String refinerPath) {
        this.pools = new ArrayList<RoutePool>();
        this.assembler = new GurobiSetPartitioningSolver(data.getNbCustomers(), true, data);
        this.msh = new MSH(assembler, GlobalParameters.THREADS);

        int numIterations = calculateNumIterations();

        addRefinerSamplingFunction(data, distances, split, numIterations, instanceName, refinerPath);

        msh.setPools(pools);
    }

    /**
     * Creates an MSH configuration for route-specific refinement
     */
    public MSHConfiguration(DataHandler data, ArrayDistanceMatrix distances, Split split,
            String instanceName, int numRoutes) {
        this.pools = new ArrayList<RoutePool>();
        this.assembler = new GurobiSetPartitioningSolver(data.getNbCustomers(), true, data);
        this.msh = new MSH(assembler, GlobalParameters.THREADS);

        int numIterations = calculateNumIterations();

        addRouteSpecificSamplingFunctions(data, distances, split, numIterations, instanceName, numRoutes);

        msh.setPools(pools);
    }

    private int calculateNumIterations() {
        int numIterations = (int) Math.ceil(GlobalParameters.MSH_NUM_ITERATIONS / 8);
        return Math.max(numIterations, 1);
    }

    private void addHighRandomizationSamplingFunctions(DataHandler data, ArrayDistanceMatrix distances,
            Split split, int numIterations) {
        // Create random generators with different seeds
        Random randomNN = new Random(GlobalParameters.SEED + 90 + 1000);
        Random randomNI = new Random(GlobalParameters.SEED + 100 + 1000);
        Random randomFI = new Random(GlobalParameters.SEED + 110 + 1000);
        Random randomBI = new Random(GlobalParameters.SEED + 120 + 1000);

        // Create and configure heuristics
        NNHeuristic nn = createNNHeuristic(distances, randomNN, GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
        InsertionHeuristic ni = createInsertionHeuristic(distances, "NEAREST_INSERTION", randomNI,
                GlobalParameters.MSH_RANDOM_FACTOR_HIGH);
        InsertionHeuristic fi = createInsertionHeuristic(distances, "FARTHEST_INSERTION", randomFI,
                GlobalParameters.MSH_RANDOM_FACTOR_HIGH);
        InsertionHeuristic bi = createInsertionHeuristic(distances, "BEST_INSERTION", randomBI,
                GlobalParameters.MSH_RANDOM_FACTOR_HIGH);

        // Create sampling functions and add to MSH
        addSamplingFunction(nn, split, numIterations, "rnn_high");
        addSamplingFunction(ni, split, numIterations, "rni_high");
        addSamplingFunction(fi, split, numIterations, "rfi_high");
        addSamplingFunction(bi, split, numIterations, "rbi_high");
    }

    private void addLowRandomizationSamplingFunctions(DataHandler data, ArrayDistanceMatrix distances,
            Split split, int numIterations) {
        // Create random generators with different seeds
        Random randomNN = new Random(GlobalParameters.SEED + 130 + 1000);
        Random randomNI = new Random(GlobalParameters.SEED + 140 + 1000);
        Random randomFI = new Random(GlobalParameters.SEED + 150 + 1000);
        Random randomBI = new Random(GlobalParameters.SEED + 160 + 1000);

        // Create and configure heuristics
        NNHeuristic nn = createNNHeuristic(distances, randomNN, GlobalParameters.MSH_RANDOM_FACTOR_LOW);
        InsertionHeuristic ni = createInsertionHeuristic(distances, "NEAREST_INSERTION", randomNI,
                GlobalParameters.MSH_RANDOM_FACTOR_LOW);
        InsertionHeuristic fi = createInsertionHeuristic(distances, "FARTHEST_INSERTION", randomFI,
                GlobalParameters.MSH_RANDOM_FACTOR_LOW);
        InsertionHeuristic bi = createInsertionHeuristic(distances, "BEST_INSERTION", randomBI,
                GlobalParameters.MSH_RANDOM_FACTOR_LOW);

        // Create sampling functions and add to MSH
        addSamplingFunction(nn, split, numIterations, "rnn_low");
        addSamplingFunction(ni, split, numIterations, "rni_low");
        addSamplingFunction(fi, split, numIterations, "rfi_low");
        addSamplingFunction(bi, split, numIterations, "rbi_low");
    }

    private void addRefinerSamplingFunction(DataHandler data, ArrayDistanceMatrix distances, Split split,
            int numIterations, String instanceName, String path) {
        Random randomNN = new Random(GlobalParameters.SEED + 90 + 1000);

        RefinerHeuristic nn = new RefinerHeuristic(distances, instanceName, path);
        nn.setRandomized(true);
        nn.setRandomGen(randomNN);
        nn.setRandomizationFactor(GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
        nn.setInitNode(0);

        addSamplingFunction(nn, split, numIterations, "rnn_refiner");
    }

    private void addRouteSpecificSamplingFunctions(DataHandler data, ArrayDistanceMatrix distances, Split split,
            int numIterations, String instanceName, int numRoutes) {
        for (int i = 0; i < numRoutes; i++) {
            Random randomNN = new Random(GlobalParameters.SEED + 90 + 1000);

            RefinerHeuristicRoutes nn = new RefinerHeuristicRoutes(distances, instanceName, i);
            nn.setRandomized(true);
            nn.setRandomGen(randomNN);
            nn.setRandomizationFactor(GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
            nn.setInitNode(0);

            addSamplingFunction(nn, split, numIterations, "rnn_route_" + i);
        }
    }

    private NNHeuristic createNNHeuristic(ArrayDistanceMatrix distances, Random random, int randomizationFactor) {
        NNHeuristic heuristic = new NNHeuristic(distances);
        heuristic.setRandomized(true);
        heuristic.setRandomGen(random);
        heuristic.setRandomizationFactor(randomizationFactor);
        heuristic.setInitNode(0);
        return heuristic;
    }

    private InsertionHeuristic createInsertionHeuristic(ArrayDistanceMatrix distances, String type,
            Random random, int randomizationFactor) {
        InsertionHeuristic heuristic = new InsertionHeuristic(distances, type);
        heuristic.setRandomized(true);
        heuristic.setRandomGen(random);
        heuristic.setRandomizationFactor(randomizationFactor);
        heuristic.setInitNode(0);
        return heuristic;
    }

    private void addSamplingFunction(Object tspHeuristic, Split split, int numIterations, String identifier) {
        OrderFirstSplitSecondHeuristic heuristic = new OrderFirstSplitSecondHeuristic(
                (core.TSPHeuristic) tspHeuristic, split);

        OrderFirstSplitSecondSampling samplingFunction = new OrderFirstSplitSecondSampling(
                heuristic, numIterations, identifier);

        RoutePool pool = new RoutePool();
        pools.add(pool);

        samplingFunction.setRoutePool(pool);
        msh.addSamplingFunction(samplingFunction);
    }

    /**
     * Creates a configuration for solution refinement using the refiner constructor
     */
    public static MSHConfiguration createRefinerConfiguration(VRPInstanceContext context) throws IOException {
        DataHandler data = context.getData();
        ArrayDistanceMatrix distances = context.getDistances();
        Split split = new split.SplitPLRP(context.getDistances(), context.getDrivingTimes(), context.getWalkingTimes(),
                context.getData());
        String path = GlobalParameters.INSTANCE_FOLDER + context.getInstanceIdentifier() + "_refiner.txt";
        return new MSHConfiguration(data, distances, split, context.getInstanceName(), path);
    }

    /**
     * Creates a configuration for route-specific refinement using the route-refiner
     * constructor
     */
    public static MSHConfiguration createRoutesRefinerConfiguration(VRPInstanceContext context, int routeCount)
            throws IOException {
        DataHandler data = context.getData();
        ArrayDistanceMatrix distances = context.getDistances();
        Split split = new split.SplitPLRP(context.getDistances(), context.getDrivingTimes(), context.getWalkingTimes(),
                context.getData());
        return new MSHConfiguration(data, distances, split, context.getInstanceName(), routeCount);
    }

    /**
     * Creates a configuration for custom cost handling using the standard
     * constructor with custom split
     */
    public static MSHConfiguration createCustomCostConfiguration(VRPInstanceContext context,
            CustomArcCostMatrix customCosts, String arcPath) throws IOException {
        DataHandler data = context.getData();
        ArrayDistanceMatrix distances = context.getDistances();
        split.SplitWithEdgeConstraints customSplit = new split.SplitWithEdgeConstraints(
                context.getDistances(), context.getDrivingTimes(), context.getWalkingTimes(),
                context.getData(), customCosts);
        return new MSHConfiguration(data, distances, customSplit);
    }

    // Getters
    public MSH getMSH() {
        return msh;
    }

    public ArrayList<RoutePool> getPools() {
        return pools;
    }

    public AssemblyFunction getAssembler() {
        return assembler;
    }
}
