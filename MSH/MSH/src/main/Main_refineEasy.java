package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import globalParameters.GlobalParametersReader;
import model.Manager;

/**
 * This class is used when we want to impose a tighter constraint
 * to see how the solution will change. For example, instead of having a
 * maximum walking distance of 2km let's use 1km, or so on..
 * 
 * We will keep the order in which the customers are visited, but for the
 * rest..we will let the algorithm decide. (The split from villegas).
 * 
 * This class relies on the fact that the MSH was already used to solve all
 * instances,
 * and that the solutions are available. To avoid errors, if the solution is not
 * available
 * it immediately stops the algorithm.
 * 
 * We apply a custom cost to flagged arcs
 * 
 * @author edouard.rabasse
 *
 */
public class Main_refineEasy {

    public static void main(String[] args) {

        // ----------------SELECT THE MAIN PARAMETERS-----------------------

        // Select the txt file, with the instance specifications: The txt files are
        // located in the experiments folder.

        String coordinatesFile = args[0]; // e.g. "Coordinates_1.txt"

        // Select the instance you want to run, (i.e., the line of the txt file): 1-79

        String arcsFile = args[1]; // e.g. "Arcs_1_1.txt"

        // Configuration file name:

        String config_file = args[2]; // e.g. default.xml

        int suffix = 1; // Default value if not provided

        if (args.length > 3) {
            suffix = Integer.parseInt(args[3]); // e.g. 1
        }

        // ----------------- MAIN LOGIC ---------------------------------

        // Create a buffered reader:

        // Store the instance name file:

        // Runs the code:

        try {

            // Loads the global parameters: some paths, the precision..

            GlobalParametersReader.initialize("./config/" + config_file);

            // Creates a Manager:

            Manager manager = new Manager();

            // Runs the MSH:

            manager.runRefineEasy(coordinatesFile, arcsFile, suffix);
            // Closes the code:

            System.exit(0);

        } catch (Exception e) {

            System.out.println("A problem running the code");
            e.printStackTrace();
        }

    }

}
