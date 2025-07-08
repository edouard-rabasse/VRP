package heuristic;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import core.InsertionHeuristic;
import core.NNHeuristic;
import core.OrderFirstSplitSecondHeuristic;
import core.RefinerHeuristic;
import core.RefinerHeuristicRoutes;
import core.RoutePool;
import core.TSPHeuristic;
import globalParameters.GlobalParameters;
import msh.MSHContext;
import msh.OrderFirstSplitSecondSampling;

public class HeuristicConfiguration {
    public enum Config {
        HIGH_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_HIGH, GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN),
        LOW_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_LOW, GlobalParameters.MSH_RANDOM_FACTOR_LOW);

        public final int randomFactor;
        public final int nnRandomFactor;

        Config(int randomFactor, int nnRandomFactor) {
            this.randomFactor = randomFactor;
            this.nnRandomFactor = nnRandomFactor;
        }
    }

    /**
     * Helper method to add a sampling function with its pool
     */
    public static void addSamplingFunction(MSHContext context, TSPHeuristic tspHeuristic, String name) {
        OrderFirstSplitSecondHeuristic heuristic = new OrderFirstSplitSecondHeuristic(tspHeuristic, context.split);
        OrderFirstSplitSecondSampling sampling = new OrderFirstSplitSecondSampling(heuristic, context.numIterations,
                name);

        RoutePool pool = new RoutePool();
        context.pools.add(pool);
        sampling.setRoutePool(pool);
        context.msh.addSamplingFunction(sampling);
    }

    /**
     * Generic method to add sampling functions with specified randomization level
     */
    public static void addSamplingFunctions(MSHContext context, HeuristicConfiguration.Config config, String suffix) {
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
     * Adds standard sampling functions (both high and low randomization)
     */
    public static void addStandardSamplingFunctions(MSHContext context) {
        addSamplingFunctions(context, HeuristicConfiguration.Config.HIGH_RANDOMIZATION, "high");
        addSamplingFunctions(context, HeuristicConfiguration.Config.LOW_RANDOMIZATION, "low");
    }

    public static void addInsertionSamplingFunction(MSHContext context, Config config,
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
     * Add refiner sampling function
     */
    public static void addRefinerSamplingFunction(MSHContext context, String path, String instance_name) {
        Random random = new Random(GlobalParameters.SEED + 90 + 1000);

        RefinerHeuristic refiner = new RefinerHeuristic(context.distances, path);
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
    public static void addRouteRefinerSamplingFunctions(MSHContext context, String path, String instance_name) {
        int numRoutes = countRoutesInFile(path);

        for (int i = 0; i < numRoutes; i++) {
            Random random = new Random(GlobalParameters.SEED + 90 + 1000);

            RefinerHeuristicRoutes refiner = new RefinerHeuristicRoutes(context.distances, instance_name, i, path);
            refiner.setRandomized(true);
            refiner.setRandomGen(random);
            refiner.setRandomizationFactor(GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN);
            refiner.setInitNode(0);

            addSamplingFunction(context, refiner, "rnn_high_" + i);
        }
    }

    /**
     * Add Nearest Neighbor sampling function
     */
    private static void addNNSamplingFunction(MSHContext context, Config config, String suffix) {
        Random random = new Random(GlobalParameters.SEED + getRandomSeed("nn", suffix));

        NNHeuristic nn = new NNHeuristic(context.distances);
        nn.setRandomized(true);
        nn.setRandomGen(random);
        nn.setRandomizationFactor(config.nnRandomFactor);
        nn.setInitNode(0);

        addSamplingFunction(context, nn, "rnn_" + suffix);
    }

    /**
     * Get random seed based on heuristic type and suffix
     */
    public static int getRandomSeed(String type, String suffix) {
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
     * Count routes in solution file
     */
    public static int countRoutesInFile(String path) {
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

}
