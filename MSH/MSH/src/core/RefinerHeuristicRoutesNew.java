package core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import globalParameters.GlobalParameters;

/**
 * TODO: Make it work (currently, gives some unfeasible solutions)
 * 
 * @author Edouard Rabasse
 * @version %I%, %G%
 * @since Jun 20 2025
 *
 */
public class RefinerHeuristicRoutesNew implements TSPHeuristic, RandomizedHeuristic {

    /**
     * The nearest neighbor finder
     */
    private final NNFinder finder;
    /**
     * The number of nodes in the instance
     */
    private final int n;
    /**
     * The random number generator
     */
    private Random rnd = null;
    /**
     * The starting node
     */
    private int initNode = 0;
    /**
     * The distance matrix
     */
    private final DistanceMatrix distances;
    /**
     * True if the heuristic runs in randomized mode and false otherwise
     */
    private boolean randomized = false;
    /**
     * Randomization factor
     */
    private int K = 1;
    /**
     * The initial route that should be completed by the heuristic
     */
    private Route initRoute = null;
    /**
     * True if the route has been initialized (by calls to {@link #setInitNode(int)
     * or #setInitRoute(Route)) and false otherwise
     */
    private boolean initialized = false;

    private String instance_id;

    private int route_id;
    private String path = null;

    /**
     * Constructs a new nearest neighbor heuristic
     * 
     * @param distances
     */
    public RefinerHeuristicRoutesNew(DistanceMatrix distances, String instance_id, int route_id) {
        this.finder = new NNFinder(distances, distances.size());
        this.n = distances.size();
        this.distances = distances;
        this.instance_id = instance_id;
        this.route_id = route_id;
        this.path = "./results/configuration1/Arcs_" + this.instance_id + "_" + GlobalParameters.SEED + ".txt";
    }

    public RefinerHeuristicRoutesNew(DistanceMatrix distances, String instance_id, int route_id, String arc_path) {
        this.finder = new NNFinder(distances, distances.size());
        this.n = distances.size();
        this.distances = distances;
        this.instance_id = instance_id;
        this.route_id = route_id;
        this.path = arc_path;
    }

    @Override
    public synchronized TSPSolution run() {

        // Option 1: Read the solution and build the tsp

        int parkingSpot = -1;
        final TSPSolution tour = this.initTour();
        double of = 0;
        List<Integer> walkingNodes = new ArrayList<>(); // Buffer for walking segment

        try {
            BufferedReader buff = new BufferedReader(new FileReader(this.path));
            String line = buff.readLine();

            while (line != null) {
                String[] parts = line.split(";");
                int tail = Integer.parseInt(parts[0]);
                int head = Integer.parseInt(parts[1]);
                int mode = Integer.parseInt(parts[2]);
                int route = Integer.parseInt(parts[3]);

                if (route == route_id) {
                    if (mode == 1) {
                        // Driving - no crossing issues, add directly
                        if (head != n) {
                            tour.add(head);
                        }
                    } else if (mode == 2) {
                        System.out.println("Processing segment: " + tail + " -> " + head + " (mode: " + mode + ")");
                        // Walking segment - handle crossings
                        if (parkingSpot == -1) {
                            // Start of walking loop
                            parkingSpot = tail;
                            tour.removeID(tail);
                            walkingNodes.clear();
                            if (head != n) {
                                walkingNodes.add(head);
                            }
                        } else if (head != parkingSpot) {
                            // Continue walking loop
                            if (head != n) {
                                walkingNodes.add(head);
                            }
                        } else if (head == parkingSpot) { // THIS WAS THE PROBLEM - removed the "mode == 2 &&" part
                            // End of walking loop - decide direction and parking placement
                            if (!walkingNodes.isEmpty()) {
                                // Try both directions and both parking positions
                                double bestCost = Double.MAX_VALUE;
                                List<Integer> bestSequence = new ArrayList<>();
                                int lastNodeInTour = tour.isEmpty() ? -1 : tour.getLastNode();

                                // Option 1: parking -> walking nodes (forward)
                                List<Integer> option1 = new ArrayList<>();
                                option1.add(parkingSpot);
                                option1.addAll(walkingNodes);
                                double cost1 = calculateSequenceCost(lastNodeInTour, option1);

                                // Option 2: parking -> walking nodes (reverse)
                                List<Integer> option2 = new ArrayList<>();
                                option2.add(parkingSpot);
                                List<Integer> reversed = new ArrayList<>(walkingNodes);
                                Collections.reverse(reversed);
                                option2.addAll(reversed);
                                double cost2 = calculateSequenceCost(lastNodeInTour, option2);

                                // Option 3: walking nodes (forward) -> parking
                                List<Integer> option3 = new ArrayList<>(walkingNodes);
                                option3.add(parkingSpot);
                                double cost3 = calculateSequenceCost(lastNodeInTour, option3);

                                // Option 4: walking nodes (reverse) -> parking
                                List<Integer> option4 = new ArrayList<>(reversed);
                                option4.add(parkingSpot);
                                double cost4 = calculateSequenceCost(lastNodeInTour, option4);

                                // Choose best option
                                if (cost1 <= cost2 && cost1 <= cost3 && cost1 <= cost4) {
                                    bestSequence = option1;
                                } else if (cost2 <= cost3 && cost2 <= cost4) {
                                    bestSequence = option2;
                                } else if (cost3 <= cost4) {
                                    bestSequence = option3;
                                } else {
                                    bestSequence = option4;
                                }

                                // Add the best sequence to tour
                                for (Integer node : bestSequence) {
                                    tour.add(node);
                                }
                            }
                            parkingSpot = -1;
                            walkingNodes.clear();
                        }
                    }
                }
                line = buff.readLine();
            }
            buff.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        tour.add(0);
        tour.setOF(of);

        return tour;
    }

    @Override
    public synchronized void setRandomized(boolean flag) {
        this.randomized = flag;
    }

    @Override
    public synchronized void setRandomGen(Random rnd) {
        this.rnd = rnd;
    }

    @Override
    public synchronized boolean isRandomized() {
        return this.randomized;
    }

    @Override
    public synchronized void setRandomizationFactor(int K) {
        this.K = K;
    }

    @Override
    public synchronized void setInitNode(int i) {
        if (this.initialized)
            throw new IllegalStateException(
                    "The heuristic has been already initialized by a call to setInitRoute(Route)");
        this.initNode = i;
        this.initialized = true;
    }

    @Override
    public synchronized void setInitRoute(Route r) {
        if (this.initialized)
            throw new IllegalStateException("The heuristic has been already initialized by a call to setInitNote(int)");
        if (r.get(0) != r.get(r.size() - 1))
            throw new IllegalArgumentException(
                    "The route must start and end at the same node. Starting and ending nodes are " + r.get(0)
                            + " and  " + r.get(r.size() - 1));
        this.initRoute = r.getCopy();
        this.initialized = true;
    }

    /**
     * Initializes the solution (i.e., an incomplete TSP tour).
     * 
     * @return an initialized TSP solution
     */
    private TSPSolution initTour() {
        final TSPSolution tour = new TSPSolution();
        // Case 1: initialized with a route
        if (this.initRoute != null) {
            initRoute.remove(initRoute.size() - 1); // the tour is still open
            tour.setRoute(initRoute);
        }
        // Case 2: initialized with a node or not initialized
        else {
            final int init;
            if (randomized && !initialized) {
                if (rnd == null) // Initialize the random number generator if needed. This will never happen if
                                 // the method remains private, but I'm thinking about pushing it up to a
                                 // superclass.
                    rnd = new Random();
                init = rnd.nextInt(n); // Randomly initialized
            } else
                init = this.initNode; // Initialized with a given node or with the default node (i.e., node 0)
            tour.add(init);
        }
        return tour;
    }

    private double calculateSequenceCost(int fromNode, List<Integer> sequence) {
        if (sequence.isEmpty())
            return 0;

        double cost = 0;
        int current = fromNode;

        for (Integer node : sequence) {
            if (current != -1) {
                cost += distances.getDistance(current, node);
            }
            current = node;
        }

        return cost;
    }
}