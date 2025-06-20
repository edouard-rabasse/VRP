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
 * MAIN DIFFERENCE: WE CHANGE EACH ROUTE !!!
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class Main_refineSolutionFixedEdges {

	public static void main(String[] args) {

		// ----------------SELECT THE MAIN PARAMETERS-----------------------

		// Select the txt file, with the instance specifications: The txt files are
		// located in the experiments folder.

		String fileName = args[0]; // e.g. ExperimentsAllSets.txt

		// Select the instance you want to run, (i.e., the line of the txt file): 1-79

		int current_instance = Integer.parseInt(args[1]); // e.g. 2

		// Configuration file name:

		String config_file = args[2]; // e.g. default.xml

		// ------------------------------------------------------------------

		// Main logic:

		// Create a buffered reader:

		try {
			BufferedReader reader = new BufferedReader(new FileReader("./experiments/" + fileName));
			int count = 0;
			String line = reader.readLine();
			count++;
			while (line != null && count < current_instance) {
				line = reader.readLine();
				count++;
			}

			args = line.split(";");
			reader.close();

		} catch (IOException e1) {
			System.out.println("The file does not exists");
			System.exit(0);
		}

		// Store the instance name file:

		String instance_identifier = args[1];

		// Runs the code:

		try {

			// Loads the global parameters: some paths, the precision..

			GlobalParametersReader.initialize("./config/" + config_file);

			// Creates a Manager:

			Manager manager = new Manager();

			// Runs the MSH:

			// manager.runRefineWithFixedEdges(instance_identifier);

			// Closes the code:

			System.exit(0);

		} catch (Exception e) {

			System.out.println("A problem running the code");
			e.printStackTrace();
		}

	}

}
