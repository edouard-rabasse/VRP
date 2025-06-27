package model;

import java.io.*;
import java.util.*;
import core.Route;
import dataStructures.DataHandler;
import core.ArrayDistanceMatrix;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import globalParameters.GlobalParameters;
import globalParameters.GlobalParametersReader;
import util.RouteFromFile;

public class SplitEasy {

    private final DataHandler data;
    private final ArrayDistanceMatrix distances;
    private final ArrayDistanceMatrix drivingTimes;
    private final ArrayDistanceMatrix walkingTimes;
    private final int nbCustomers;

    public SplitEasy(String instanceIdentifier) throws IOException {
        // Same initialization as RouteFromFile
        this.data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instanceIdentifier);
        this.distances = new DepotToCustomersDistanceMatrix(data);
        this.drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        this.walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);
        this.nbCustomers = data.getNbCustomers();
    }

    public ArrayList<Route> modifyRoutesFromFile(String arcFilePath, String outputPath) throws IOException {
        // Use RouteFromFile's parsing method directly
        Map<Integer, List<RouteFromFile.Arc>> routeArcsMap = RouteFromFile.parseArcFile(arcFilePath);

        // Modify each route
        Map<Integer, List<RouteFromFile.Arc>> modifiedRouteArcsMap = new HashMap<>();
        ArrayList<Route> modifiedRoutes = new ArrayList<>();

        for (Map.Entry<Integer, List<RouteFromFile.Arc>> entry : routeArcsMap.entrySet()) {
            int routeId = entry.getKey();
            List<RouteFromFile.Arc> arcs = entry.getValue();

            // Modify this route's arcs
            List<RouteFromFile.Arc> modifiedArcs = modifyRouteArcs(arcs, routeId);
            modifiedRouteArcsMap.put(routeId, modifiedArcs);

            // Build Route object using RouteFromFile's method
            Route route = RouteFromFile.buildRouteFromArcs(modifiedArcs, data, distances, drivingTimes, walkingTimes,
                    routeId);
            modifiedRoutes.add(route);
        }

        // Write modified arcs to file
        writeModifiedArcsToFile(modifiedRouteArcsMap, outputPath);

        return modifiedRoutes;
    }

    /**
     * Modify arcs of a single route to respect walking constraints
     */
    private List<RouteFromFile.Arc> modifyRouteArcs(List<RouteFromFile.Arc> originalArcs, int routeId) {
        List<RouteFromFile.Arc> modifiedArcs = new ArrayList<>();
        List<RouteFromFile.Arc> sortedArcs = RouteFromFile.sortArcsIntoPath(new ArrayList<>(originalArcs));

        if (sortedArcs.isEmpty()) {
            return modifiedArcs;
        }
        int currentPosition = nbCustomers + 1; // Start at depot
        int currentParkingSpot = -1; // Start at depot
        double routeWalkingDistance = 0.0;
        int lastNode = 0;
        Set<Integer> visitedNodes = new HashSet<>(); // Track visited nodes to avoid repeating arcs

        for (int i = 0; i < sortedArcs.size(); i++) {
            RouteFromFile.Arc arc = sortedArcs.get(i);

            if (arc.tail != currentPosition) {

                // We need to connect to this arc's tail first
                if (arc.mode == 2) { // If this was supposed to be walking
                    // Drive to the tail if we're not already there
                    modifiedArcs.add(new RouteFromFile.Arc(currentPosition, arc.tail, 1)); // Drive
                    currentPosition = arc.tail;
                    currentParkingSpot = arc.tail; // New parking spot
                    routeWalkingDistance = 0.0; // Reset walking distance
                } else {
                    // For driving arcs, just update the tail
                    arc = new RouteFromFile.Arc(currentPosition, arc.head, 1);
                }
            }

            if (arc.mode == 2) { // Walking arc
                // Check if this walking arc violates constraints
                double arcWalkingDistance = distances.getDistance(
                        arc.tail % (nbCustomers + 1),
                        arc.head % (nbCustomers + 1));

                if (currentParkingSpot == -1) {
                    currentParkingSpot = arc.tail; // Start at depot
                }
                double returnDistance = distances.getDistance(currentParkingSpot % (nbCustomers + 1),
                        arc.head % (nbCustomers + 1));

                if (routeWalkingDistance + arcWalkingDistance
                        + returnDistance > GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT ||
                        returnDistance > GlobalParameters.MAX_WD_B2P) {
                    // add the return to the parking spot
                    if (currentPosition != currentParkingSpot) {
                        modifiedArcs.add(new RouteFromFile.Arc(currentPosition, currentParkingSpot, 2));
                        currentPosition = currentParkingSpot;
                    }

                    // Add driving arc to destination only if we haven't visited it already
                    if (!visitedNodes.contains(arc.head)) {
                        modifiedArcs.add(new RouteFromFile.Arc(currentPosition, arc.head, 1));
                        currentPosition = arc.head;
                        currentParkingSpot = arc.head; // Update parking spot
                        routeWalkingDistance = 0.0; // Reset walking distance
                        visitedNodes.add(arc.head);
                    } else {
                    }
                } else {
                    // Valid walking arc, add to modified arcs
                    modifiedArcs.add(new RouteFromFile.Arc(arc.tail, arc.head, 2));
                    routeWalkingDistance += arcWalkingDistance;
                    currentPosition = arc.head;
                    visitedNodes.add(arc.head);
                }
            } else { // Driving arc
                     // Check if we need to return to parking spot before this driving arc
                     // if (currentParkingSpot != -1 && routeWalkingDistance > 0) {
                     // double returnDistance = distances.getDistance(currentParkingSpot, lastNode);
                     // if (returnDistance <= GlobalParameters.MAX_WD_B2P) {
                     // // Add return to parking spot
                     // modifiedArcs.add(new RouteFromFile.Arc(lastNode, currentParkingSpot, 2));
                     // routeWalkingDistance = 0.0; // Reset walking distance
                     // }
                     // }

                // Add the driving arc
                if (!visitedNodes.contains(arc.head)) {
                    modifiedArcs.add(new RouteFromFile.Arc(currentPosition, arc.head, 1));
                    currentPosition = arc.head;
                    currentParkingSpot = arc.head; // Update parking spot
                    visitedNodes.add(arc.head);
                } else {
                }
            }
        }
        return modifiedArcs;
    }

    /**
     * Write modified arcs to file
     */
    private void writeModifiedArcsToFile(Map<Integer, List<RouteFromFile.Arc>> modifiedRouteArcsMap, String outputPath)
            throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            for (Map.Entry<Integer, List<RouteFromFile.Arc>> entry : modifiedRouteArcsMap.entrySet()) {
                for (RouteFromFile.Arc arc : entry.getValue()) {
                    writer.println(arc.tail + ";" + arc.head + ";" + arc.mode + ";" + entry.getKey());
                }
            }
        }
    }

    // Usage example
    public static void main(String[] args) {
        try {
            String instanceFile = "Coordinates_5.txt";
            String inputArcFile = "results/configuration1/Arcs_5_1.txt";
            String outputArcFile = "results/configuration8/Arcs_5_1_modified.txt";

            String config_path = "config/configuration7.xml";
            GlobalParametersReader.initialize(config_path);
            SplitEasy modifier = new SplitEasy(instanceFile);
            ArrayList<Route> modifiedRoutes = modifier.modifyRoutesFromFile(inputArcFile, outputArcFile);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
