package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import globalParameters.GlobalParameters;

/**
 * Utility class for handling route file operations and analysis.
 * This class provides methods to read route files and extract information for
 * refinement operations.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class RouteFileUtils {

    /**
     * Counts the number of routes in a solution file
     */
    public static int countRoutesInFile(String filePath) throws IOException {
        if (!new File(filePath).exists()) {
            throw new IOException("Route file does not exist: " + filePath);
        }

        int numRoutes = 0;
        int routeInitial = -1;

        try (BufferedReader buff = new BufferedReader(new FileReader(filePath))) {
            String line = buff.readLine();

            while (line != null) {
                String[] parts = line.split(";");
                if (parts.length >= 4) {
                    int route = Integer.parseInt(parts[3]);
                    if (routeInitial != route) {
                        numRoutes++;
                        routeInitial = route;
                    }
                }
                line = buff.readLine();
            }
        }

        return numRoutes;
    }

    /**
     * Gets the standard route file path for a given instance and seed
     */
    public static String getStandardRouteFilePath(String instanceName) {
        return "./results/configuration1/Arcs_" + instanceName + "_" + GlobalParameters.SEED + ".txt";
    }

    /**
     * Gets the refiner file path for a given instance
     */
    public static String getRefinerFilePath(String instanceIdentifier) {
        return GlobalParameters.INSTANCE_FOLDER + instanceIdentifier + "_refiner.txt";
    }

    /**
     * Checks if a route file exists and is readable
     */
    public static boolean isRouteFileValid(String filePath) {
        File file = new File(filePath);
        return file.exists() && file.isFile() && file.canRead();
    }

    /**
     * Gets the global path for an arc file
     */
    public static String getGlobalArcPath(String arcPath) {
        return GlobalParameters.RESULT_FOLDER + arcPath;
    }

    /**
     * Gets the comparison file path for calculating original costs
     */
    public static String getComparisonFilePath(String instanceName) {
        return GlobalParameters.COMPARISON_FOLDER + "Arcs_" + instanceName + "_" + 1 + ".txt";
    }
}
