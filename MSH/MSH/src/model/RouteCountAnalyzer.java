package model;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import globalParameters.GlobalParameters;

/**
 * Utility class to analyze route counts from solution files.
 * This is used for route refinement operations where we need to know
 * how many routes are in the existing solution.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class RouteCountAnalyzer {

    private final String instanceName;

    public RouteCountAnalyzer(String instanceName) {
        this.instanceName = instanceName;
    }

    /**
     * Gets the number of routes from the default solution file path
     */
    public int getRouteCount() throws IOException {
        String path = "./results/configuration1/Arcs_" + instanceName + "_" + GlobalParameters.SEED + ".txt";
        return getRouteCount(path);
    }

    /**
     * Gets the number of routes from a specific solution file
     */
    public int getRouteCount(String filePath) throws IOException {
        int numRoutes = 0;

        try (BufferedReader buff = new BufferedReader(new FileReader(filePath))) {
            String line = buff.readLine();
            int routeInitial = -1;

            while (line != null) {
                String[] parts = line.split(";");
                int route = Integer.parseInt(parts[3]);
                if (routeInitial != route) {
                    numRoutes++;
                    routeInitial = route;
                }
                line = buff.readLine();
            }
        } catch (IOException e) {
            System.out.println("Error trying to read the solution and creating the tsp associated with it");
            System.out.println("We will stop the code here");
            e.printStackTrace();
            System.exit(0);
        }

        return numRoutes;
    }
}
