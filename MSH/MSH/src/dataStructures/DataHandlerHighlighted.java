//DataHandlerHighlighted.java : this file is used to read instances where :
// coordinates are node,x,y,coordtype(0 or 1)
// arcs are node1,node2,dist,arctype(0 or 1)

package dataStructures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import distanceMatrices.ArcModificationMatrix;

/**
 * This class stores the main parameters of the current instance, as the number
 * customers, the service times
 * and the customers' coordinates.
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class DataHandlerHighlighted extends DataHandler {

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

	/**
	 * Coordinates type: 0 for normal, 1 for needing change
	 */
	private ArrayList<Integer> coord_types;

	// METHODS:

	/**
	 * This method creates a new DataHandler.
	 * 
	 * @param path: path to the instance.dat file
	 * @throws IOException
	 */
	public DataHandlerHighlighted(String path) throws IOException {
		super(path);

		// Read the coordinates of the nodes:

		// 0. Creates a buffered reader:

		BufferedReader buff = new BufferedReader(new FileReader(path));

		// 1. Depot information

		service_times = new ArrayList<Double>();
		x_coors = new ArrayList<Double>();
		y_coors = new ArrayList<Double>();
		coord_types = new ArrayList<Integer>();
		String line = buff.readLine();
		nbCustomers = 0;
		while (line != null) {
			String[] parts = line.split(",");
			line = buff.readLine();
			if (line == null) {
				x_coors.add(Double.parseDouble(parts[1]));
				y_coors.add(Double.parseDouble(parts[2]));
				service_times.add(Double.parseDouble(parts[3]));
				coord_types.add(Integer.parseInt(parts[4]));
			} else {
				nbCustomers += 1;
			}

		}

		buff.close();

		// 2. Customers information:

		buff = new BufferedReader(new FileReader(path));

		for (int i = 0; i < nbCustomers; i++) {
			line = buff.readLine();
			String[] parts = line.split(",");
			x_coors.add(Double.parseDouble(parts[1]));
			y_coors.add(Double.parseDouble(parts[2]));
			service_times.add(Double.parseDouble(parts[3]));
			coord_types.add(Integer.parseInt(parts[4]));
		}

		// 3. Close the buffered reader:

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
	 * @return the coordinates types (1 if in a modification zone)
	 */
	public ArrayList<Integer> getCoord_types() {
		return coord_types;
	}

	/**
	 * @param service_times the service_times to set
	 */
	public void setService_times(ArrayList<Double> service_times) {
		this.service_times = service_times;
	}

	/**
	 * @param coord_types the coord_types to set
	 */
	public void setCoord_types(ArrayList<Integer> coord_types) {
		this.coord_types = coord_types;
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
