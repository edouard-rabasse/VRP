package util;

import java.io.File;
import java.io.PrintWriter;
import core.Route;
import msh.AssemblyFunction;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import validation.RouteConstraintValidator;
import core.ArrayDistanceMatrix;
import core.RouteAttribute;

/**
 * Utility class for printing solutions with cost analysis
 * 
 * @author Edouard Rabasse
 */
public class SolutionPrinter {

    public static void printSolutionWithCostAnalysis(AssemblyFunction assembler, DataHandler data,
            String instanceName, int suffix,
            ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes,
            RouteConstraintValidator routeValidator) {
        printSolutionWithCostAnalysis(assembler, data, instanceName, suffix, originalDistances, originalWalkingTimes,
                routeValidator, null);
    }

    public static void printSolutionWithCostAnalysis(AssemblyFunction assembler, DataHandler data,
            String instanceName, int suffix,
            ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes,
            RouteConstraintValidator routeValidator, Double realCost) {
        // Call the overloaded method with RealCost
        printSolutionWithCostAnalysis(assembler, data, instanceName, suffix, originalDistances, originalWalkingTimes,
                routeValidator, realCost, null);
    }

    public static void printSolutionWithCostAnalysis(AssemblyFunction assembler, DataHandler data,
            String instanceName, int suffix,
            ArrayDistanceMatrix originalDistances,
            ArrayDistanceMatrix originalWalkingTimes,
            RouteConstraintValidator routeValidator,
            Double RealCost, Double easyCost) {

        String pathArcs = GlobalParameters.RESULT_FOLDER + "Arcs_" + instanceName + "_" + suffix + ".txt";
        String pathCosts = GlobalParameters.RESULT_FOLDER + "CostAnalysis_" + instanceName + "_" + suffix + ".txt";

        try {

            createDirectoriesIfNeeded(pathArcs);
            createDirectoriesIfNeeded(pathCosts);
            PrintWriter pwArcs = new PrintWriter(new File(pathArcs));
            PrintWriter pwCosts = new PrintWriter(new File(pathCosts));

            // Headers
            pwCosts.println("Route;OldCost;NewCost;EasyCost;Penalty;PenaltyPercentage;Chain;Valid;numberOfViolations");

            System.out.println("-----------------------------------------------");
            System.out.println("SOLUTION WITH COST ANALYSIS");
            System.out.println("Total custom cost: " + assembler.objectiveFunction);

            double totalOldCost = 0.0;
            double totalNewCost = 0.0;

            int routeCounter = 0;
            int numberOfViolations = 0;
            for (Route route : assembler.solution) {

                // Calculate cost breakdown
                CostBreakdown breakdown = CostCalculator.calculateCostBreakdown(
                        route, data, originalDistances, originalWalkingTimes, null);

                totalOldCost += breakdown.getRealCost();
                totalNewCost += breakdown.getCustomCost();
                boolean isValidRoute = routeValidator.validateRoute(route).isValid;
                numberOfViolations += routeValidator.validateRoute(route).getViolationCount();

                // Print route analysis
                // String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
                System.out.printf("Route %d: Old=%.2f, New=%.2f, Penalty=%.2f (%.1f%%)%n Valid=%b",
                        routeCounter, breakdown.getRealCost(), breakdown.getCustomCost(),
                        breakdown.getPenalty(), breakdown.getPenaltyPercentage(), isValidRoute);

                // // Save to cost analysis file
                // pwCosts.printf("%d;%.2f;%.2f;%.2f;%.1f;%s%n",
                // routeCounter, breakdown.getOldCost(), breakdown.getNewCost(),
                // breakdown.getPenalty(), breakdown.getPenaltyPercentage(), chain);

                // Generate arcs (existing logic)
                generateArcsFromRoute(route, data, pwArcs, routeCounter);

                System.out.println(
                        "\t\t Route " + routeCounter + ": " + route.getAttribute(RouteAttribute.CHAIN) + " - Cost: "
                                + route.getAttribute(RouteAttribute.COST) + " - Duration: "
                                + route.getAttribute(RouteAttribute.DURATION) + " - Service time: "
                                + route.getAttribute(RouteAttribute.SERVICE_TIME));

                routeCounter++;
            }
            boolean isValid = routeValidator.validateGlobalConstraints(assembler.solution);

            if (RealCost != null) {
                // If RealCost is provided, use it for totalOldCost, meaning that we compare
                // with another configuration

                totalNewCost = totalOldCost;
                totalOldCost = RealCost;
            }
            double easyCostValue = easyCost != null ? easyCost : 0.0;

            // Summary
            double totalPenalty = totalNewCost - totalOldCost;
            System.out.println("-----------------------------------------------");
            System.out.printf("TOTALS: Real=%.2f, Custom=%.2f, EasyCost=%.2f, Penalty=%.2f (%.1f%%)%n",
                    totalOldCost, totalNewCost, easyCostValue, totalPenalty,
                    totalOldCost > 0 ? (totalPenalty / totalOldCost * 100) : 0);

            pwCosts.printf("TOTAL;%.2f;%.2f;%.2f;%.2f;%.1f;%s;%b;%d%n",
                    totalOldCost, totalNewCost, easyCostValue, totalPenalty,
                    totalOldCost > 0 ? (totalPenalty / totalOldCost * 100) : 0, "", isValid, numberOfViolations);

            pwArcs.close();
            pwCosts.close();

        } catch (Exception e) {
            System.out.println("Error printing solution with cost analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void generateArcsFromRoute(Route route, DataHandler data, PrintWriter pwArcs, int routeCounter) {
        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
        chain = chain.replace(" 0 ", " " + (data.getNbCustomers() + 1) + " ");
        chain = chain.replace("CD", "" + (data.getNbCustomers() + 1));
        chain = chain.replace(" ", "");
        String[] parts = chain.split("[||]");

        for (int pos = 0; pos < parts.length; pos++) {

            if (parts[pos].length() > 0) {

                // Only driving:

                if (parts[pos].contains("->") && !parts[pos].contains("---")) {

                    parts[pos] = parts[pos].replace("->", ";");
                    String[] arcParts = parts[pos].split("[;]");
                    for (int arc = 0; arc < arcParts.length - 1; arc++) {
                        if (!arcParts[arc].equals(arcParts[arc + 1])) {

                            pwArcs.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + 1 + ";" + routeCounter);
                        }
                    }

                }

                // Only walking:

                if (!parts[pos].contains("->") && parts[pos].contains("---")) {

                    parts[pos] = parts[pos].replace("---", ";");
                    String[] arcParts = parts[pos].split("[;]");
                    for (int arc = 0; arc < arcParts.length - 1; arc++) {
                        pwArcs.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + 2 + ";" + routeCounter);
                    }

                }

                // Mix between driving and walking:

                if (parts[pos].contains("->") && parts[pos].contains("---")) {

                    parts[pos] = parts[pos].replace("---", ";");
                    parts[pos] = parts[pos].replace("->", ":");

                    int posInString = 0;
                    String tail = "";
                    String head = "";
                    int mode = -1;
                    int lastPos = -1;
                    while (posInString < parts[pos].length()) {
                        if (mode == -1) {
                            if (parts[pos].charAt(posInString) == ':') {
                                mode = 1;
                                lastPos = posInString;
                            }
                            if (parts[pos].charAt(posInString) == ';') {
                                mode = 2;
                                lastPos = posInString;
                            }
                        } else {
                            if (parts[pos].charAt(posInString) == ':') {
                                if (!tail.equals(head)) {
                                    pwArcs.println(tail + ";" + head + ";" + mode + ";" + routeCounter);
                                }
                                posInString = lastPos;
                                mode = -1;
                                tail = "";
                                head = "";
                            }
                            if (parts[pos].charAt(posInString) == ';') {
                                if (!tail.equals(head)) {
                                    pwArcs.println(tail + ";" + head + ";" + mode + ";" + routeCounter);
                                }
                                posInString = lastPos;
                                mode = -1;
                                tail = "";
                                head = "";
                            }
                        }

                        if (mode == -1 && !(parts[pos].charAt(posInString) == ':')
                                && !(parts[pos].charAt(posInString) == ';')) {
                            tail += parts[pos].charAt(posInString);
                        } else {
                            if (!(parts[pos].charAt(posInString) == ':')
                                    && !(parts[pos].charAt(posInString) == ';')) {
                                head += parts[pos].charAt(posInString);
                            }
                        }

                        posInString++;
                    }
                    if (!tail.equals(head)) {
                        pwArcs.println(tail + ";" + head + ";" + mode + ";" + routeCounter);
                    }

                }

            }

        }
    }

    /**
     * Creates parent directories for a file path if they don't exist
     * 
     * @param filePath The file path
     */
    private static void createDirectoriesIfNeeded(String filePath) {
        File file = new File(filePath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            boolean created = parentDir.mkdirs();
            if (!created) {
                System.err.println("Warning: Could not create directory: " + parentDir.getAbsolutePath());
            }
        }
    }

}