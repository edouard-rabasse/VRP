package model;

import java.io.File;
import java.io.PrintWriter;

import core.Route;
import core.RouteAttribute;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.MSH;
import util.SolutionPrinter;
import validation.RouteConstraintValidator;
import util.RouteFromFile;
import core.ArrayDistanceMatrix;

/**
 * Handles the output and reporting of VRP solutions.
 * This class encapsulates solution printing, summary generation, and cost
 * analysis.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class SolutionReporter {

    private final String instanceName;
    private final double initializationTime;

    public SolutionReporter(String instanceName, double initializationTime) {
        this.instanceName = instanceName;
        this.initializationTime = initializationTime;
    }

    /**
     * Prints solution with default seed suffix
     */
    public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data) {
        printSolution(msh, assembler, data, GlobalParameters.SEED);
    }

    /**
     * Prints solution with custom suffix
     */
    public void printSolution(MSH msh, AssemblyFunction assembler, DataHandler data, int suffix) {
        String pathArcs = GlobalParameters.RESULT_FOLDER + "Arcs_" + instanceName + "_" + suffix + ".txt";

        System.out.println("Saving the solution in: " + pathArcs);

        try (PrintWriter pwArcs = new PrintWriter(new File(pathArcs))) {
            System.out.println("-----------------------------------------------");
            System.out.println("Total cost: " + assembler.objectiveFunction);
            System.out.println("Routes:");

            int counter = 0;
            for (Route r : assembler.solution) {
                processRoute(r, data, pwArcs, counter);
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
     * Prints solution summary to console and file
     */
    public void printSummary(MSH msh, AssemblyFunction assembler, DataHandler data,
            double samplingTime, double assemblyTime) {
        String path = GlobalParameters.RESULT_FOLDER + "Summary_" + instanceName + "_" + GlobalParameters.SEED + ".txt";

        try (PrintWriter pw = new PrintWriter(new File(path))) {
            // Write summary data
            writeSummaryData(pw, data, assembler, msh, samplingTime, assemblyTime);

            // Print to console
            printSummaryToConsole(data, assembler, msh, samplingTime, assemblyTime);

        } catch (Exception e) {
            System.out.println("Mistake printing the summary");
        }
    }

    /**
     * Prints solution with cost analysis and validation
     */
    public void printSolutionWithAnalysis(AssemblyFunction assembler, DataHandler data, int suffix,
            ArrayDistanceMatrix distances, ArrayDistanceMatrix walkingTimes,
            String instanceIdentifier, double originalCost) {
        try {
            RouteConstraintValidator validator = new RouteConstraintValidator(instanceIdentifier,
                    "./config/configuration7.xml");

            SolutionPrinter.printSolutionWithCostAnalysis(assembler, data, instanceName, suffix,
                    distances, walkingTimes, validator, originalCost);
        } catch (java.io.IOException e) {
            System.out.println("Error initializing RouteConstraintValidator: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void processRoute(Route route, DataHandler data, PrintWriter pwArcs, int counter) {
        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
        chain = chain.replace(" 0 ", " " + (data.getNbCustomers() + 1) + " ");
        chain = chain.replace("CD", "" + (data.getNbCustomers() + 1));
        chain = chain.replace(" ", "");
        String[] parts = chain.split("[||]");

        for (String part : parts) {
            if (part.length() > 0) {
                processRoutePart(part, pwArcs, counter);
            }
        }
    }

    private void processRoutePart(String part, PrintWriter pwArcs, int counter) {
        if (part.contains("->") && !part.contains("---")) {
            // Only driving
            processDrivingSegment(part, pwArcs, counter);
        } else if (!part.contains("->") && part.contains("---")) {
            // Only walking
            processWalkingSegment(part, pwArcs, counter);
        } else if (part.contains("->") && part.contains("---")) {
            // Mixed driving and walking
            processMixedSegment(part, pwArcs, counter);
        }
    }

    private void processDrivingSegment(String part, PrintWriter pwArcs, int counter) {
        part = part.replace("->", ";");
        String[] arcParts = part.split("[;]");
        for (int arc = 0; arc < arcParts.length - 1; arc++) {
            if (!arcParts[arc].equals(arcParts[arc + 1])) {
                pwArcs.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + 1 + ";" + counter);
            }
        }
    }

    private void processWalkingSegment(String part, PrintWriter pwArcs, int counter) {
        part = part.replace("---", ";");
        String[] arcParts = part.split("[;]");
        for (int arc = 0; arc < arcParts.length - 1; arc++) {
            pwArcs.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + 2 + ";" + counter);
        }
    }

    private void processMixedSegment(String part, PrintWriter pwArcs, int counter) {
        part = part.replace("---", ";");
        part = part.replace("->", ":");

        int posInString = 0;
        String tail = "";
        String head = "";
        int mode = -1;
        int lastPos = -1;

        while (posInString < part.length()) {
            char c = part.charAt(posInString);

            if (mode == -1) {
                if (c == ':') {
                    mode = 1;
                    lastPos = posInString;
                } else if (c == ';') {
                    mode = 2;
                    lastPos = posInString;
                }
            } else {
                if (c == ':' || c == ';') {
                    if (!tail.equals(head)) {
                        pwArcs.println(tail + ";" + head + ";" + mode + ";" + counter);
                    }
                    posInString = lastPos;
                    mode = -1;
                    tail = "";
                    head = "";
                }
            }

            if (mode == -1 && c != ':' && c != ';') {
                tail += c;
            } else if (c != ':' && c != ';') {
                head += c;
            }

            posInString++;
        }

        if (!tail.equals(head)) {
            pwArcs.println(tail + ";" + head + ";" + mode + ";" + counter);
        }
    }

    private void printRouteInfo(Route route, int counter) {
        System.out.println("\t\t Route " + counter + ": " + route.getAttribute(RouteAttribute.CHAIN) +
                " - Cost: " + route.getAttribute(RouteAttribute.COST) +
                " - Duration: " + route.getAttribute(RouteAttribute.DURATION) +
                " - Service time: " + route.getAttribute(RouteAttribute.SERVICE_TIME));
    }

    private void writeSummaryData(PrintWriter pw, DataHandler data, AssemblyFunction assembler,
            MSH msh, double samplingTime, double assemblyTime) {
        pw.println("Instance;" + instanceName);
        pw.println("Instance_n;" + data.getNbCustomers());
        pw.println("Seed;" + GlobalParameters.SEED);
        pw.println("InitializationTime(s);" + initializationTime);
        pw.println("TotalTime(s);" + (samplingTime + assemblyTime));
        pw.println("TotalDistance;" + assembler.objectiveFunction);
        pw.println("Iterations;" + msh.getNumberOfIterations());
        pw.println("SizeOfPool;" + msh.getPoolSize());
        pw.println("SamplingTime(s);" + samplingTime);
        pw.println("AssemblyTime(s);" + assemblyTime);
    }

    private void printSummaryToConsole(DataHandler data, AssemblyFunction assembler,
            MSH msh, double samplingTime, double assemblyTime) {
        System.out.println("-----------------------------------------------");
        System.out.println("Instance: " + instanceName);
        System.out.println("Instance_n: " + data.getNbCustomers());
        System.out.println("Seed: " + GlobalParameters.SEED);
        System.out.println("InitializationTime(s): " + initializationTime);
        System.out.println("TotalTime(s): " + (samplingTime + assemblyTime));
        System.out.println("TotalDistance: " + assembler.objectiveFunction);
        System.out.println("Iterations: " + msh.getNumberOfIterations());
        System.out.println("SizeOfPool: " + msh.getPoolSize());
        System.out.println("SamplingTime(s): " + samplingTime);
        System.out.println("AssemblyTime(s): " + assemblyTime);
    }
}
