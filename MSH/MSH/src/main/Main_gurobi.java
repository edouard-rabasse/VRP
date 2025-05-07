package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import globalParameters.GlobalParametersReader;
import model.Manager;

/**
 * This class runs the MSH procedure for solving the Park-and-loop routing problem.
 * The user can select the instance.
 * 
 * Instances:
 * 	VRPTR instances proposed by Coindreau et al. (2019).
 * 
 * If you want to change more parameters, you can modify the "default.xml" file.
 * For example:
 * 	-Printing useful information on console
 * 	-Modifying the number of iterations for each TSP heuristic..
 * 	Among others..
 * 
 * This code requires gurobi! Very important !
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class Main_gurobi {

	public static void main(String[] args) {
		
	// ----------------SELECT THE MAIN PARAMETERS-----------------------
	
			
			
			String fileName = args[0];        
			// String fileName = "ExperimentsAllSets.txt";        // e.g. ExperimentsAllSets.txt
			int current_instance = Integer.parseInt(args[1]); // e.g. 2
			// int current_instance = 200; // e.g. 2
			String config_file = args[2];
			// String config_file = "default.xml"; // e.g. default.xml
			
		// ------------------------------------------------------------------	
			
		// Main logic:
			
			
			// Create a buffered reader:
			
			try {
				BufferedReader reader = new BufferedReader(new FileReader("./experiments/"+fileName));
				int count = 0;
				String line = reader.readLine();
				count++;
				while(line != null && count < current_instance) {
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
				
					GlobalParametersReader.initialize("./config/"+config_file);
					
				// Creates a Manager:
					
					Manager manager = new Manager();
					
				// Runs the MSH:
					
					manager.runMSH_gurobi(instance_identifier);
					
				// Closes the code:
					
					System.exit(0);
				
			}catch(Exception e) {
				
				System.out.println("A problem running the code");
				e.printStackTrace();
			} 
	
	}	
	
}
