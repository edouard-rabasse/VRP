package util;

import core.Route;
import core.RouteAttribute;
import dataStructures.DataHandler;
import core.ArrayDistanceMatrix;
import globalParameters.GlobalParameters;
import distanceMatrices.CustomArcCostMatrix;

/**
 * Utility class for calculating different types of costs for routes
 */
public class CostCalculator {

    /**
     * Calculate the real cost of a route using original distances
     * (without custom costs or penalties)
     * 
     * @author Edouard Rabasse
     */
    public static double calculateRealRouteCost(Route route, DataHandler data,
            ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes) {

        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);

        double realCost = GlobalParameters.FIXED_COST; // Base fixed cost
        double drivingCost = 0.0;

        // Parse segments
        String[] segments = chain.split("\\|\\|");

        for (String segment : segments) {
            if (segment.contains("->") && segment.contains("---")) {
                // Mixed segment - both driving and walking
                realCost += parseMixedSegment(segment, originalDistances, originalWalkingTimes, data);

            } else if (segment.contains("->") && !segment.contains("---")) {
                // Pure driving segment
                realCost += parseDrivingSegment(segment, originalDistances);

            } else if (!segment.contains("->") && segment.contains("---")) {
                // Pure walking segment - no cost in PLRP model
            }
        }

        return realCost;
    }

    /**
     * Calculate cost breakdown: real vs custom vs fixed arcs
     */
    public static CostBreakdown calculateCostBreakdown(Route route, DataHandler data,
            ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes,
            CustomArcCostMatrix customCosts) {

        double realCost = calculateRealRouteCost(route, data, originalDistances, originalWalkingTimes);
        double customCost = (Double) route.getAttribute(RouteAttribute.COST);

        return new CostBreakdown(realCost, customCost, customCost - realCost);
    }

    /**
     * Parse a mixed segment containing both driving (-->) and walking (---) arcs
     * Example: "CD -> 0 -> 3 --- 7 --- 17 --- 3"
     */
    private static double parseMixedSegment(String segment, ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes, DataHandler data) {

        double segmentCost = 0.0;

        // Strategy: Parse character by character to detect mode transitions
        String currentNode = "";
        String previousNode = "";
        int currentMode = -1; // -1=unknown, 1=driving, 2=walking

        int i = 0;
        while (i < segment.length()) {
            char c = segment.charAt(i);

            // Detect mode transitions
            if (c == '-') {
                // Check if it's -> (driving) or --- (walking)
                if (i + 1 < segment.length() && segment.charAt(i + 1) == '>') {
                    // Driving mode: ->
                    if (!previousNode.isEmpty() && !currentNode.isEmpty()) {
                        // Process the arc with the previous mode if it exists
                        if (currentMode == 1) {
                            segmentCost += addDrivingArc(previousNode, currentNode, originalDistances, data);
                        } else if (currentMode == 2) {
                            addWalkingArc(previousNode, currentNode); // No cost for walking
                        }
                    }

                    previousNode = currentNode.trim();
                    currentNode = "";
                    currentMode = 1; // Next arc will be driving
                    i += 2; // Skip '->'
                    continue;

                } else if (i + 2 < segment.length() &&
                        segment.charAt(i + 1) == '-' && segment.charAt(i + 2) == '-') {
                    // Walking mode: ---
                    if (!previousNode.isEmpty() && !currentNode.isEmpty()) {
                        // Process the arc with the previous mode
                        if (currentMode == 1) {
                            segmentCost += addDrivingArc(previousNode, currentNode, originalDistances, data);
                        } else if (currentMode == 2) {
                            addWalkingArc(previousNode, currentNode); // No cost for walking
                        }
                    }

                    previousNode = currentNode.trim();
                    currentNode = "";
                    currentMode = 2; // Next arc will be walking
                    i += 3; // Skip '---'
                    continue;
                }
            }

            // Skip whitespace around separators
            if (c == ' ' && (currentNode.isEmpty() ||
                    (i + 1 < segment.length() && segment.charAt(i + 1) == '-'))) {
                i++;
                continue;
            }

            // Accumulate node characters
            if (c != ' ') {
                currentNode += c;
            }

            i++;
        }

        // Process the final arc
        if (!previousNode.isEmpty() && !currentNode.isEmpty()) {
            if (currentMode == 1) {
                segmentCost += addDrivingArc(previousNode, currentNode.trim(), originalDistances, data);
            } else if (currentMode == 2) {
                addWalkingArc(previousNode, currentNode.trim()); // No cost for walking
            }
        }

        return segmentCost;
    }

    /**
     * Add a driving arc and return its cost
     */
    private static double addDrivingArc(String fromStr, String toStr,
            ArrayDistanceMatrix originalDistances, DataHandler data) {

        // Convert node names to integers
        int from = convertNodeToInt(fromStr);
        int to = convertNodeToInt(toStr);

        if (from != -1 && to != -1 && from != to) {

            double arcCost = originalDistances.getDistance(from, to) * GlobalParameters.VARIABLE_COST;

            return arcCost;
        }

        return 0.0;
    }

    /**
     * Add a walking arc (no cost, just logging)
     */
    private static void addWalkingArc(String fromStr, String toStr) {

    }

    /**
     * Convert node string to integer, handling special cases like "CD"
     */
    private static int convertNodeToInt(String nodeStr) {
        if (nodeStr == null || nodeStr.isEmpty()) {
            return -1;
        }

        nodeStr = nodeStr.trim();

        if (nodeStr.equals("CD")) {
            // CD represents depot - convert to depot number
            return 0; // or 0, depending on your depot numbering
        }

        try {
            return Integer.parseInt(nodeStr);
        } catch (NumberFormatException e) {

            return -1;
        }
    }

    private static double parseDrivingSegment(String segment, ArrayDistanceMatrix originalDistances) {
        double segmentCost = 0.0;

        String[] parts = segment.split("->");
        for (int i = 0; i < parts.length - 1; i++) {
            String fromStr = parts[i].trim();
            String toStr = parts[i + 1].trim();

            if (!fromStr.equals("CD")) {
                int from = convertNodeToInt(fromStr);
                int to = convertNodeToInt(toStr);

                segmentCost += originalDistances.getDistance(from, to) * GlobalParameters.VARIABLE_COST;
            }
        }

        return segmentCost;
    }
}