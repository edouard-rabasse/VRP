package main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

public class CreateInstances2 {

	
    private static final DecimalFormat decimalFormat = new DecimalFormat("0.000");
    private static final Random random = new Random(); //Setting a seed
    
    public static void main(String[] args) {
    	
    	double maxX = 10;
    	double maxY = 10;
    	int numClients = 50;
    	int initialInstanceID = 301;
    	int finalInstanceID = 1000;
    	
   
    	for(int i = initialInstanceID; i <= finalInstanceID; i++) {
    		generateInstance(numClients,maxX,maxY,""+i);
    	}
    }

    private static void generateInstance(int numClients, double maxX, double maxY, String instanceID) {
        String fileName = "Coordinates_" + instanceID + ".txt";
        String folderPath = "./instances";

        // Ensure the instances folder exists
        File folder = new File(folderPath);
        if (!folder.exists()) {
            folder.mkdir();
        }

        File file = new File(folderPath + File.separator + fileName);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            for (int i = 1; i <= numClients; i++) {
                double x = formatDouble(random.nextDouble() * maxX);
                double y = formatDouble(random.nextDouble() * maxY);
                int serviceTime = 20 + random.nextInt(16); // 20 to 35 inclusive

                writer.write(i + "," + decimalFormat.format(x) + "," + decimalFormat.format(y) + "," + serviceTime);
                writer.newLine();
            }
            writer.write((numClients+1)+ "," + decimalFormat.format(5) + "," + decimalFormat.format(5) + "," + 0);
            writer.write("");
        } catch (IOException e) {
            System.err.println("Error writing instance file: " + e.getMessage());
        }
    }

    private static double formatDouble(double value) {
        // Round to 3 decimal places
	        return Math.round(value * 1000.0) / 1000.0;
	    }
	
}
