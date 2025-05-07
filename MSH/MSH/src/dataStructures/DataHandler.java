package dataStructures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * This class stores the main parameters of the current instance, as the number customers, the service times
 * and the customers' coordinates.
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class DataHandler {
	
	/**
	 * Number of customers
	 */
	
	private int nbCustomers;
	
	/**
	 * Service times
	 */
	private ArrayList<Double> service_times;
	
	
	/**
	 * X coordinates
	 */
	private ArrayList<Double> x_coors;
	
	
	/**
	 * Y coordinates
	 */
	private ArrayList<Double> y_coors;
	
	
	// METHODS:
	
	
	/**
	 * This method creates a new DataHandler.
	 * 
	 * @param path: path to the instance.dat file
	 * @throws IOException
	 */
	public DataHandler(String path) throws IOException{
		
		// Read the coordinates of the nodes:
		
			//0. Creates a buffered reader:
			
				BufferedReader buff = new BufferedReader(new FileReader(path));
		
			//1. Depot information
				
				service_times = new ArrayList<Double>();
				x_coors = new ArrayList<Double>();
				y_coors = new ArrayList<Double>();
				String line = buff.readLine();
				nbCustomers = 0;
				while(line != null) {
					String[] parts = line.split(",");
					line = buff.readLine();
					if(line == null) {
						x_coors.add(Double.parseDouble(parts[1]));
						y_coors.add(Double.parseDouble(parts[2]));
						service_times.add(Double.parseDouble(parts[3]));
					}else {
						nbCustomers += 1;
					}
					
				}
				
				buff.close();
				
			//2. Customers information:	
			
				buff = new BufferedReader(new FileReader(path));
				
				for(int i = 0;i < nbCustomers;i++) {
					line = buff.readLine();
					String[] parts = line.split(",");
					x_coors.add(Double.parseDouble(parts[1]));
					y_coors.add(Double.parseDouble(parts[2]));
					service_times.add(Double.parseDouble(parts[3]));
				}
				
			//3. Close the buffered reader:
				
				buff.close();
				
	}


	/**
	 * @return the nbCustomers
	 */
	public int getNbCustomers() {
		return nbCustomers;
	}


	/**
	 * @param nbCustomers the nbCustomers to set
	 */
	public void setNbCustomers(int nbCustomers) {
		this.nbCustomers = nbCustomers;
	}


	/**
	 * @return the service_times
	 */
	public ArrayList<Double> getService_times() {
		return service_times;
	}


	/**
	 * @param service_times the service_times to set
	 */
	public void setService_times(ArrayList<Double> service_times) {
		this.service_times = service_times;
	}


	/**
	 * @return the x_coors
	 */
	public ArrayList<Double> getX_coors() {
		return x_coors;
	}


	/**
	 * @param x_coors the x_coors to set
	 */
	public void setX_coors(ArrayList<Double> x_coors) {
		this.x_coors = x_coors;
	}


	/**
	 * @return the y_coors
	 */
	public ArrayList<Double> getY_coors() {
		return y_coors;
	}


	/**
	 * @param y_coors the y_coors to set
	 */
	public void setY_coors(ArrayList<Double> y_coors) {
		this.y_coors = y_coors;
	}

	
	
	
}
