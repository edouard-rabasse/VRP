package util;

import java.io.*;
import java.lang.reflect.Array;
import java.util.*;
import core.Route;
import core.RoutePool;
import core.RouteAttribute;
import core.JVRAEnv;
import dataStructures.DataHandler;
import core.ArrayDistanceMatrix;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import globalParameters.GlobalParameters;
import globalParameters.GlobalParametersReader;

/**
 * Utility class to create Route objects from arc files
 */
public class RouteFromFile {

    public static RoutePool routePool = new RoutePool();

    public static int nbCustomers = 0;

    /**
     * Create routes from an arc file
     * Format: tail;head;mode;route_id
     * Example: 3;1;1;0 (from node 3 to node 1, driving mode, route 0)
     */
    public static ArrayList<Route> createRoutesFromFile(String arcFilePath, String instanceIdentifier)
            throws IOException {

        // Load instance data
        DataHandler data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instanceIdentifier);

        // Load distance matrices
        ArrayDistanceMatrix distances = new DepotToCustomersDistanceMatrix(data);
        ArrayDistanceMatrix drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        ArrayDistanceMatrix walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);

        nbCustomers = data.getNbCustomers();

        // Group arcs by route ID
        Map<Integer, List<Arc>> routeArcsMap = parseArcFile(arcFilePath);

        // Create Route objects
        ArrayList<Route> routes = new ArrayList<>();

        for (Map.Entry<Integer, List<Arc>> entry : routeArcsMap.entrySet()) {
            int routeId = entry.getKey();
            List<Arc> arcs = entry.getValue();

            Route route = buildRouteFromArcs(arcs, data, distances, drivingTimes, walkingTimes, routeId);
            routes.add(route);
        }

        return routes;
    }

    /**
     * Parse the arc file and group arcs by route ID
     */
    public static Map<Integer, List<Arc>> parseArcFile(String arcFilePath) throws IOException {
        Map<Integer, List<Arc>> routeArcsMap = new HashMap<>();

        BufferedReader reader = new BufferedReader(new FileReader(arcFilePath));
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("//"))
                continue; // Skip empty lines and comments

            String[] parts = line.split(";");
            if (parts.length >= 4) {
                try {
                    int tail = Integer.parseInt(parts[0]);
                    int head = Integer.parseInt(parts[1]);
                    int mode = Integer.parseInt(parts[2]); // 1=driving, 2=walking
                    int routeId = Integer.parseInt(parts[3]);

                    Arc arc = new Arc(tail, head, mode);
                    routeArcsMap.computeIfAbsent(routeId, k -> new ArrayList<>()).add(arc);

                } catch (NumberFormatException e) {
                    System.err.println("Warning: Invalid line format: " + line);
                }
            }
        }
        reader.close();

        return routeArcsMap;
    }

    public static double getTotalAttribute(RouteAttribute attribute, String arcFilePath, String instanceIdentifier)
            throws IOException {
        ArrayList<Route> routes = createRoutesFromFile(arcFilePath, instanceIdentifier);
        double total = 0.0;
        for (Route route : routes) {
            Object value = route.getAttribute(attribute);
            if (value instanceof Number) {
                total += ((Number) value).doubleValue();
            } else {
                System.err.println("Warning: Attribute " + attribute + " is not a number in route " + route);
            }
        }

        return total;
    }

    /**
     * Build a Route object from a list of arcs
     */
    public static Route buildRouteFromArcs(List<Arc> arcs, DataHandler data,
            ArrayDistanceMatrix distances, ArrayDistanceMatrix drivingTimes,
            ArrayDistanceMatrix walkingTimes, int routeId) {

        // Create Route using factory
        Route route = JVRAEnv.getRouteFactory().buildRoute();

        // Sort arcs to form a path
        List<Arc> sortedArcs = sortArcsIntoPath(new ArrayList<>(arcs));

        // Build node sequence
        LinkedHashSet<Integer> nodeSequence = new LinkedHashSet<>();
        for (Arc arc : sortedArcs) {
            nodeSequence.add(arc.tail);
            nodeSequence.add(arc.head);
        }

        // Add nodes to route
        for (Integer node : nodeSequence) {
            route.add(node);
        }

        // Build chain string
        String chain = buildChainString(sortedArcs, data);

        // Calculate metrics
        RouteMetrics metrics = calculateRouteMetrics(sortedArcs, data, distances, drivingTimes, walkingTimes);

        // Set route attributes
        route.setAttribute(RouteAttribute.CHAIN, chain);
        route.setAttribute(RouteAttribute.COST, metrics.totalCost);
        route.setAttribute(RouteAttribute.DURATION, metrics.totalDuration);
        route.setAttribute(RouteAttribute.SERVICE_TIME, metrics.serviceTime);
        route.setAttribute(RouteAttribute.DISTANCE, metrics.totalDistance);

        return route;
    }

    /**
     * Sort arcs to form a continuous path
     */
    public static List<Arc> sortArcsIntoPath(List<Arc> arcs) {
        if (arcs.isEmpty())
            return arcs;

        List<Arc> sortedArcs = new ArrayList<>();

        // Find starting arc (from depot or lowest node)
        Arc startArc = arcs.get(0);

        sortedArcs.add(startArc);
        arcs.remove(startArc);

        // Build path by following connections
        while (!arcs.isEmpty()) {
            int currentHead = sortedArcs.get(sortedArcs.size() - 1).head;

            Arc nextArc = arcs.stream()
                    .filter(arc -> arc.tail == currentHead)
                    .findFirst()
                    .orElse(null);

            if (nextArc != null) {
                sortedArcs.add(nextArc);
                arcs.remove(nextArc);
            } else {
                // No connection found, add remaining arcs as is
                sortedArcs.addAll(arcs);
                break;
            }
        }

        return sortedArcs;
    }

    /**
     * Build chain string from arcs (like "CD -> 1 -> 2 --- 3 --- CD")
     */
    public static String buildChainString(List<Arc> sortedArcs, DataHandler data) {
        if (sortedArcs.isEmpty())
            return "";

        List<String> segments = new ArrayList<>();
        StringBuilder currentSegment = new StringBuilder();
        int lastMode = -1;

        for (int i = 0; i < sortedArcs.size(); i++) {
            Arc arc = sortedArcs.get(i);
            String tailStr = convertNodeToString(arc.tail, data);
            String headStr = convertNodeToString(arc.head, data);

            if (lastMode != arc.mode) {
                // Mode change - start new segment
                if (currentSegment.length() > 0) {
                    segments.add(currentSegment.toString());
                }
                currentSegment = new StringBuilder();
                currentSegment.append(tailStr);
                lastMode = arc.mode;
            }

            // Add arc to current segment
            if (arc.mode == 1) { // driving
                currentSegment.append(" -> ").append(headStr);
            } else { // walking
                currentSegment.append(" --- ").append(headStr);
            }
        }

        // Add final segment
        if (currentSegment.length() > 0) {
            segments.add(currentSegment.toString());
        }

        return String.join(" || ", segments);
    }

    /**
     * Calculate route metrics
     */
    public static RouteMetrics calculateRouteMetrics(List<Arc> arcs, DataHandler data,
            ArrayDistanceMatrix distances, ArrayDistanceMatrix drivingTimes,
            ArrayDistanceMatrix walkingTimes) {

        RouteMetrics metrics = new RouteMetrics();
        Set<Integer> visitedNodes = new HashSet<>();

        for (Arc arc : arcs) {
            double arcDistance = distances.getDistance(arc.tail % (nbCustomers + 1), arc.head % (nbCustomers + 1));

            if (arc.mode == 1) { // driving
                metrics.drivingTime += drivingTimes.getDistance(arc.tail % (nbCustomers + 1),
                        arc.head % (nbCustomers + 1));
                metrics.totalCost += arcDistance * GlobalParameters.VARIABLE_COST;
                metrics.drivingDistance += arcDistance;
            } else if (arc.mode == 2) { // walking
                metrics.walkingTime += walkingTimes.getDistance(arc.tail % (nbCustomers + 1),
                        arc.head % (nbCustomers + 1));
                metrics.walkingDistance += arcDistance;
            }

            // Add service time once per node
            if (!visitedNodes.contains(arc.tail) && arc.tail > 0) {
                metrics.serviceTime += data.getService_times().get(arc.tail % (nbCustomers + 1));
                visitedNodes.add(arc.tail);
            }
            if (!visitedNodes.contains(arc.head) && arc.head > 0) {
                metrics.serviceTime += data.getService_times().get(arc.head % (nbCustomers + 1));
                visitedNodes.add(arc.head);
            }
        }

        // Add fixed cost and parking time
        metrics.totalCost += GlobalParameters.FIXED_COST;
        metrics.parkingTime = GlobalParameters.PARKING_TIME_MIN;

        metrics.totalDuration = metrics.walkingTime + metrics.drivingTime +
                metrics.serviceTime + metrics.parkingTime;
        metrics.totalDistance += metrics.drivingDistance + metrics.walkingDistance;

        return metrics;
    }

    /**
     * Convert node number to string representation
     */
    public static String convertNodeToString(int node, DataHandler data) {
        if (node == 0 || node == data.getNbCustomers() + 1) {
            return "CD"; // Depot
        }
        return String.valueOf(node);
    }

    // Helper classes
    public static class Arc {
        public int tail, head, mode;

        public Arc(int tail, int head, int mode) {
            this.tail = tail;
            this.head = head;
            this.mode = mode;
        }

        @Override
        public String toString() {
            return String.format("%d->%d(mode:%d)", tail, head, mode);
        }
    }

    public static class RouteMetrics {
        double totalCost = 0.0;
        double totalDuration = 0.0;
        double drivingTime = 0.0;
        double walkingTime = 0.0;
        double walkingDistance = 0.0;
        double drivingDistance = 0.0;
        double serviceTime = 0.0;
        double parkingTime = 0.0;
        double totalDistance = 0.0;
    }

    // Usage example
    public static void main(String[] args) {
        try {
            int instance_number = args[0] != null ? Integer.parseInt(args[0]) : 1;
            String arcFile = "results/configuration1/Arcs_" + instance_number + "_1.txt";
            String instance = "Coordinates_" + instance_number + ".txt";
            GlobalParametersReader.initialize("config/configuration1.xml");

            ArrayList<Route> routes = createRoutesFromFile(arcFile, instance);

            System.out.println("Created " + routes.size() + " routes from file:");
            for (int i = 0; i < routes.size(); i++) {
                Route route = routes.get(i);
                System.out.println("\nRoute " + i + ":");
                System.out.println("  Nodes: " + route.getRoute());
                System.out.println("  Chain: " + route.getAttribute(RouteAttribute.CHAIN));
                System.out.println("  Cost: " + route.getAttribute(RouteAttribute.COST));
                System.out.println("  Duration: " + route.getAttribute(RouteAttribute.DURATION) + " min");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}