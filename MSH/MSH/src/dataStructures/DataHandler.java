package dataStructures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;

import core.TSPSolution;
import globalParameters.GlobalParameters;

/**
 * This class stores the main parameters of the current instance, as the number
 * customers, the service times
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

	private Hashtable<Integer, Integer> mapping;

	// METHODS:

	/**
	 * This method creates a new DataHandler.
	 * 
	 * @param path: path to the instance.dat file
	 * @throws IOException
	 */
	public DataHandler(String path) throws IOException {

		// Read the coordinates of the nodes:

		// 0. Creates a buffered reader:

		BufferedReader buff = new BufferedReader(new FileReader(path));

		// 1. Depot information

		service_times = new ArrayList<Double>();
		x_coors = new ArrayList<Double>();
		y_coors = new ArrayList<Double>();
		String line = buff.readLine();
		nbCustomers = 0;
		while (line != null) {
			String[] parts = line.split(",");
			line = buff.readLine();
			if (line == null) {
				x_coors.add(Double.parseDouble(parts[1]));
				y_coors.add(Double.parseDouble(parts[2]));
				service_times.add(Double.parseDouble(parts[3]));
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
		}

		// 3. Close the buffered reader:

		buff.close();

		// // 4. Set the mapping : each customer is mapped to its index in the list
		// mapping = new Hashtable<Integer, Integer>();
		// for (int i = 1; i <= nbCustomers; i++) {
		// mapping.put(i, i);
		// }

		// this.setMapping(mapping);

	}

	/**
	 * This method creates a new DataHandler.
	 * 
	 * @param coord_path: path to the instance.dat file
	 * @param arc_path:   path to the arcs.txt file
	 * @param route_id:   id of the route
	 * @param n:          number of nodes (including the depot)
	 * @throws IOException
	 */
	public DataHandler(String coord_path, String arc_path, int route_id, int n) throws IOException {

		// Identify the nodes in the route:

		Hashtable<Integer, Integer> mapping = new Hashtable<Integer, Integer>();

		// Option 1: Read the solution and build the tsp

		String arcs_path = arc_path;
		int parkingSpot = -1;
		Integer[] tour = new Integer[n];
		for (int i = 0; i < n; i++) {
			tour[i] = 0;
		}

		BufferedReader buff_arcs = new BufferedReader(new FileReader(arcs_path));

		String line = buff_arcs.readLine();

		while (line != null) {
			String[] parts = line.split(";");
			int tail = Integer.parseInt(parts[0]);
			int head = Integer.parseInt(parts[1]);
			int mode = Integer.parseInt(parts[2]);
			int route = Integer.parseInt(parts[3]);

			if (route == route_id) {

				if (mode == 1) {

					if (head != n) {

						tour[head - 1] = 1;

					}

				}
				if (mode == 2 && parkingSpot == -1) {

					parkingSpot = tail;

					if (head != n) {

						tour[head - 1] = 1;

					}
				}

				else if (mode == 2 && head != parkingSpot && parkingSpot != -1) {

					if (head != n) {

						tour[head - 1] = 1;

					}

				}

				else if (mode == 2 && head == parkingSpot) {

					parkingSpot = -1;

				}
			}

			line = buff_arcs.readLine();

		}

		buff_arcs.close();

		// Read the coordinates of the nodes:

		// 0. Creates a buffered reader:

		BufferedReader buff = new BufferedReader(new FileReader(coord_path));

		// 1. Depot information

		service_times = new ArrayList<Double>();
		x_coors = new ArrayList<Double>();
		y_coors = new ArrayList<Double>();
		line = buff.readLine();
		nbCustomers = 0;
		while (line != null) {
			String[] parts = line.split(",");
			line = buff.readLine();
			if (line == null) {
				x_coors.add(Double.parseDouble(parts[1]));
				y_coors.add(Double.parseDouble(parts[2]));
				service_times.add(Double.parseDouble(parts[3]));
			} else {
				nbCustomers += 1;

			}
		}

		buff.close();

		// 2. Customers information:

		buff = new BufferedReader(new FileReader(coord_path));

		int counter = 1;
		for (int i = 0; i < nbCustomers; i++) {
			line = buff.readLine();
			String[] parts = line.split(",");
			if (tour[i] == 1) {
				x_coors.add(Double.parseDouble(parts[1]));
				y_coors.add(Double.parseDouble(parts[2]));
				service_times.add(Double.parseDouble(parts[3]));
				mapping.put(counter, (i + 1));

				counter++;
			}
		}

		// 3. Close the buffered reader:

		buff.close();

		this.setMapping(mapping);

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

	public Hashtable<Integer, Integer> getMapping() {
		return mapping;
	}

	public void setMapping(Hashtable<Integer, Integer> mapping) {
		this.mapping = mapping;
	}

}
