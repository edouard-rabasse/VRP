package split;

import java.util.ArrayList;
import java.util.HashMap;

import core.DistanceMatrix;
import core.JVRAEnv;
import core.Route;
import core.RouteAttribute;
import core.Solution;
import core.Split;
import core.TSPSolution;
import core.VRPSolution;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import pulseStructures.PulseHandler;
import pulseStructures.PulseHandlerCC;
import pulseStructures.PulseNode;
import distanceMatrices.CustomArcCostMatrix;

/**
 * Implements the split procedure as described in Cabrera et al. (2022).
 * With a twist to add custom cost.
 * 
 */
public class SplitWithEdgeConstraints implements Split {

	/**
	 * The distance matrix
	 */
	private final DistanceMatrix distances;

	/**
	 * The driving times matrix
	 */
	private final DistanceMatrix driving_times;

	/**
	 * The walking times matrix
	 */
	private final DistanceMatrix walking_times;

	/**
	 * The customer demands
	 */
	private final DataHandler data;

	/**
	 * The custom arc costs matrix
	 */
	private final CustomArcCostMatrix customArcCosts;

	/**
	 * This method creates a new instance of the HpMP split algorithm
	 * 
	 * @param distances
	 * @param m
	 * @param M
	 */
	public SplitWithEdgeConstraints(DistanceMatrix distances, DistanceMatrix d_times, DistanceMatrix w_times,
			DataHandler data, CustomArcCostMatrix customArcCosts) {
		this.distances = distances;
		this.driving_times = d_times;
		this.walking_times = w_times;
		this.data = data;
		this.customArcCosts = customArcCosts;
	}

	@Override
	public Solution split(TSPSolution r) {

		// Remove the first and last node of the tsp (the depot)

		r.remove(0);
		r.remove(r.size() - 1);

		// Number of nodes:

		int N = r.size() + 1;

		// Initializes a set of hashmaps to store information about the routes built:

		HashMap<String, String> routeStrings = new HashMap<String, String>();
		HashMap<String, Double> routeTimes = new HashMap<String, Double>();
		HashMap<String, Double> routeCosts = new HashMap<String, Double>();

		// Initialize labels for the shortest path:

		int[] P = new int[N]; // The predecesor labels
		double[] V = new double[N]; // The shortest path labels: Cost
		double[] T = new double[N]; // The shortest path labels: Total time
		for (int i = 1; i < N; i++) {
			V[i] = Double.MAX_VALUE;
			T[i] = Double.MAX_VALUE;
		}

		// 1. First loop: All split nodes

		for (int i = 0; i < N - 1; i++) { // (Tail)

			// Check a route where only the current node is visited:
			// Use custom arc costs to calculate the cost and time of the route

			int j = i + 1; // (Head)

			int customer = r.get(j - 1);

			// (1) Compute total “driving cost” of (0→cust→0), using CUSTOM costs if
			// available:
			double singleDrivingCost = 0.0;

			// --- depot → cust:
			if (customArcCosts.hasCustomCost(0, customer, 1)) {
				singleDrivingCost += customArcCosts.getCustomCost(0, customer, 1);
			} else {
				singleDrivingCost += distances.getDistance(0, customer) * GlobalParameters.VARIABLE_COST;
			}

			// --- cust → depot:
			if (customArcCosts.hasCustomCost(customer, 0, 1)) {
				singleDrivingCost += customArcCosts.getCustomCost(customer, 0, 1);
			} else {
				singleDrivingCost += distances.getDistance(customer, 0) * GlobalParameters.VARIABLE_COST;
			}

			double singleRouteTime = driving_times.getDistance(0, customer)
					+ data.getService_times().get(customer)
					+ GlobalParameters.PARKING_TIME_MIN
					+ driving_times.getDistance(customer, 0);

			double candidateCost = V[i] + GlobalParameters.FIXED_COST + singleDrivingCost;

			if (candidateCost < V[j]) {
				V[j] = candidateCost; // Update the shortest path label
				P[j] = i;
				// Add's the route:
				routeStrings.put((i + ";" + j), "CD -> " + customer);
				routeTimes.put((i + ";" + j), singleRouteTime);
				routeCosts.put((i + ";" + j), candidateCost);

			}

			// The first head:

			j = i + 2; // (Head)

			// Creates a pulse handler (an empty board for the pulse):

			PulseHandlerCC pH = new PulseHandlerCC(data, distances, driving_times, walking_times, r, customArcCosts);

			// A counter to know if the network must be created from scratch or not:

			int cont = 1;
			boolean stopLoop = false; // Should we continue building arcs using this tail node?

			// 2. Second loop: all possible head nodes:

			r.insert(0, i); // Inserts the depot into the tsp tour position

			while (!stopLoop && j < N) {

				stopLoop = true;

				double[] newWeights = new double[2];

				if (cont == 1) {

					newWeights = calculateDRWW(i, j + 1, pH, r);
					cont++;

				} else {
					newWeights = calculateDRWWRep(i, j + 1, pH, r);

				}

				//// We check if the pulse was able to build a solution:

				if (newWeights[1] < Double.POSITIVE_INFINITY) {

					// We must continue:

					stopLoop = false;

					// Check if we should update the shortest path labels:

					if (newWeights[0] + V[i] < V[j]) {

						// Adds the route:

						routeStrings.put((i + ";" + j), recoverRoute(pH.getPath(), r, (i + ";" + j)));
						routeTimes.put((i + ";" + j), newWeights[1]);
						routeCosts.put((i + ";" + j), newWeights[0]);

						// Updates labels:

						V[j] = newWeights[0] + V[i];
						P[j] = i;

					}

				}

				// Update j
				j += 1;

			}

			r.remove(i);

		}

		return extractRoutes(P, V, T, r, N, routeStrings, routeTimes, routeCosts);
	}

	/**
	 * Extracts the routes from the labels, builds a solution, and evaluates the
	 * solution
	 * 
	 * @param P   the predecessors
	 * @param V   the shortest path labels
	 * @param tsp the TSP tour
	 * @return a solution with the routes in the optimal partition of the TSP tour
	 */
	private VRPSolution extractRoutes(int[] P, double[] V, double[] T, TSPSolution tsp, int N,
			HashMap<String, String> routeStrings, HashMap<String, Double> routeTimes,
			HashMap<String, Double> routeCosts) {

		VRPSolution s = new VRPSolution();
		double of = 0;
		int head = N - 1;
		while (head != 0) {
			int tail = P[head]; // The tail of the arc representing the route being currently built

			// Create the new route

			Route r = JVRAEnv.getRouteFactory().buildRoute();
			r.add(0);
			double service_time = 0;
			for (int i = tail; i < head; i++) {
				int node = tsp.get(i);
				service_time += data.getService_times().get(node);
				r.add(node);
			}

			r.add(0);
			r.setAttribute(RouteAttribute.SERVICE_TIME, service_time);
			r.setAttribute(RouteAttribute.COST, routeCosts.get(tail + ";" + head));

			r.setAttribute(RouteAttribute.DURATION, routeTimes.get(tail + ";" + head));
			r.setAttribute(RouteAttribute.CHAIN, routeStrings.get(tail + ";" + head));

			// Add the route to the solution

			s.addRoute(r);
			of += routeCosts.get(tail + ";" + head);
			head = P[head];

		}
		s.setOF(of);
		return s;
	}

	/**
	 * This method creates the auxiliary graph for the pulse algorithm and runs the
	 * pulse
	 * 
	 * @param i  tail node on the split graph
	 * @param j  head node on the split graph
	 * @param pH board for the pulse
	 * @return the weights of the minimum cost path that satisfies the time
	 *         constraint
	 */
	public double[] calculateDRWW(int i, int j, PulseHandler pH, TSPSolution tspTour) {

		// Creates the weights

		double[] newWeights = new double[2];

		// Create nodes:

		for (int l = i + 1; l <= j; l++) { // rows
			for (int m = i + 1; m <= j; m++) { // columns
				if (l <= m) {
					pH.addNode(l + "," + m);
				}
			}
		}

		// Modifies the labels of the initial node (Where the shortest path begins)

		// pH.getNodes().get((i+1)+","+(i+1)).setLabelsCD(distances.getDistance(tspTour.get(i),0)*GlobalParameters.VARIABLE_COST,driving_times.getDistance(tspTour.get(i),0));
		// //We set the first move from the CD !!!!!
		pH.getNodes().get((i + 1) + "," + (i + 1)).setLabelsCD(0, 0); // We set the first move from the CD !!!!!

		// Create arcs: (Note that we do it like on the bellman ford procedure !! (Going
		// by layers)

		developFirst(i + 1, i + 1, j, i + 1, pH, tspTour);

		for (int m = i + 2; m < j; m++) { // columns
			for (int l = i + 1; l <= j; l++) { // rows
				if (l <= m) {
					develop(l, m, j, i + 1, pH, tspTour);
				}
			}
		}

		// This method runs the pulse algorithm

		pH.runPulse(i, j, GlobalParameters.ROUTE_DURATION_LIMIT, GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT,
				((i + 1) + "," + (i + 1)), 1);

		// Saves the solution of the pulse

		newWeights[0] = pH.getPrimalBound() + GlobalParameters.FIXED_COST;
		newWeights[1] = pH.getTimeStar();

		return newWeights;
	}

	/**
	 * This method creates the auxiliary graph for the pulse algorithm and runs the
	 * pulse.
	 * However, it only adds a layer of nodes !! Not all of them..
	 * 
	 * @param i  tail node on the split graph
	 * @param j  head node on the split graph
	 * @param pH board for the pulse
	 * @return the weights of the minimum cost path that satisfies the time
	 *         constraint
	 */
	public double[] calculateDRWWRep(int i, int j, PulseHandler pH, TSPSolution tspTour) {

		// Creates the weights

		double[] newWeights = new double[2];

		// Create only nodes that are new !

		for (int l = i + 1; l <= j; l++) {
			for (int m = j; m <= j; m++) {
				pH.addNode(l + "," + m);
			}
		}

		// Modifies the labels of the initial node (Where the shortest path begins)
		// (Maybe this is not needed..)

		// pH.getNodes().get((i+1)+","+(i+1)).setLabelsCD(distances.getDistance(tspTour.get(i),0)*GlobalParameters.VARIABLE_COST,driving_times.getDistance(tspTour.get(i),0));
		// //We set the first move from the CD !!!!!
		pH.getNodes().get((i + 1) + "," + (i + 1)).setLabelsCD(0, 0); // We set the first move from the CD !!!!!

		// Create new arcs:

		developFirst2(i + 1, i + 1, j, i + 1, pH, tspTour);

		for (int m = i + 2; m < j; m++) {
			for (int l = i + 1; l < j; l++) {
				if (l <= m) {
					develop2(l, m, j, i + 1, pH, tspTour);
				}
			}
		}

		// This method runs the pulse algorithm

		pH.runPulse(i, j, GlobalParameters.ROUTE_DURATION_LIMIT, GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT,
				((i + 1) + "," + (i + 1)), 1);

		// Saves the solution of the pulse

		newWeights[0] = pH.getPrimalBound() + GlobalParameters.FIXED_COST;
		newWeights[1] = pH.getTimeStar();

		return newWeights;
	}

	/**
	 * This method creates the arcs for the pulse board
	 * 
	 * @param l    vertical coordinate - parking spot
	 * @param m    horizontal coordinate - client
	 * @param last maximum horizontal coordinate
	 * @param beg  initial vertical coordinate
	 */
	public void developFirst(int l, int m, int last, int beg, PulseHandler pH, TSPSolution tspTour) {

		// Initialize:

		double walkTime = 0;
		double servTime = 0;

		servTime += data.getService_times().get(tspTour.get(m - 1));

		int w = tspTour.get(m); // Client!
		int u = w;

		// Main loop:

		for (int j = m + 1; j <= last; j++) {
			int v = tspTour.get(j - 1); // Client!
			walkTime += walking_times.getDistance(u, v); // Going forward (between clients)
			servTime += data.getService_times().get(v);

			u = v;
			for (int p = beg; p <= last; p++) {
				if (p <= j) {

					// Update locations:

					int L = tspTour.get(p - 1); // Parking spot!

					int k2 = tspTour.get(l - 1); // Parking spot!

					// Update walking time

					double walkTimeA = walkTime + walking_times.getDistance(L, w) + walking_times.getDistance(v, L); // Movimiento
																														// de
																														// ps2
																														// a
																														// un
																														// cliente
																														// +
																														// De
																														// vuelta
																														// desde
																														// el
																														// last
																														// client
																														// al
																														// ps2
					double tourService = servTime;
					if (l == m && l != p) {
						tourService -= data.getService_times().get(tspTour.get(m - 1));
					}

					// Check if we must add the arc:

					if (walkTimeA + tourService <= GlobalParameters.SUBTOUR_TIME_LIMIT
							&& walkTimeA <= GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT) {

						// Parking time:

						double parkTime = 0;
						if (l != p) {
							parkTime = GlobalParameters.PARKING_TIME_MIN;
						}

						// Driving time and distances

						double divTimeA = driving_times.getDistance(k2, L); // Ir del ps1 al ps2
						double divDisA = distances.getDistance(k2, L); // Ir del ps1 al ps2

						double totalCost = calculateTotalCost(k2, L, w, v, m, j, tspTour);

						// Update total time
						double totTimeA = walkTimeA + divTimeA + parkTime + servTime;

						pH.addArc(l + "," + m, p + "," + j, totalCost, totTimeA, walkTimeA);

						PulseNode nodeI = pH.getNodes().get(l + "," + m);
						PulseNode nodeF = pH.getNodes().get(p + "," + j);

						if (totTimeA + nodeI.getMinTime()[1] < nodeF.getMinTime()[1]) {
							nodeF.getMinTime()[0] = totalCost + nodeI.getMinTime()[0];
							nodeF.getMinTime()[1] = totTimeA + nodeI.getMinTime()[1];
							nodeF.getMinTime()[2] = walkTimeA + nodeI.getMinTime()[2];
						}
						if (totalCost + nodeI.getMinCost()[0] < nodeF.getMinCost()[0]) {
							nodeF.getMinCost()[0] = totalCost + nodeI.getMinCost()[0];
							nodeF.getMinCost()[1] = totTimeA + nodeI.getMinCost()[1];
							nodeF.getMinCost()[2] = walkTimeA + nodeI.getMinCost()[2];
						}

						if (walkTimeA + nodeI.getMinDist()[2] < nodeF.getMinDist()[2]) {
							nodeF.getMinDist()[0] = totalCost + nodeI.getMinDist()[0];
							nodeF.getMinDist()[1] = totTimeA + nodeI.getMinDist()[1];
							nodeF.getMinDist()[2] = walkTimeA + nodeI.getMinDist()[2];
						}

					}
				}
			}
		}
	}

	/**
	 * This method creates the arc for the pulse board
	 * 
	 * @param l    vertical coordinate - parking spot
	 * @param m    horizontal coordinate - client
	 * @param last maximum horizontal coordinate
	 * @param beg  initial vertical coordinate
	 */
	public void develop(int l, int m, int last, int beg, PulseHandler pH, TSPSolution tspTour) {

		// Initialize:

		double walkTime = 0;
		double servTime = 0;

		int w = tspTour.get(m); // Client!
		int u = w;

		// Main loop:

		for (int j = m + 1; j <= last; j++) {
			int v = tspTour.get(j - 1); // Client!
			walkTime += walking_times.getDistance(u, v); // Going forward (between clients)
			servTime += data.getService_times().get(v);
			u = v;
			for (int p = beg; p <= last; p++) {
				if (p <= j) {

					// Update locations:

					int L = tspTour.get(p - 1); // Parking spot!

					int k2 = tspTour.get(l - 1); // Parking spot!

					// Update walking time

					double walkTimeA = walkTime + walking_times.getDistance(L, w) + walking_times.getDistance(v, L); // Movimiento
																														// de
																														// ps2
																														// a
																														// un
																														// cliente
																														// +
																														// De
																														// vuelta
																														// desde
																														// el
																														// last
																														// client
																														// al
																														// ps2

					double tourService = servTime;
					if (l == m && l != p) {
						tourService -= data.getService_times().get(tspTour.get(m - 1));
					}

					// Check if we must add the arc

					if (walkTimeA + tourService <= GlobalParameters.SUBTOUR_TIME_LIMIT
							&& walkTimeA <= GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT) {

						// Parking time:

						double parkTime = 0;
						if (l != p) {
							parkTime = GlobalParameters.PARKING_TIME_MIN;
						}

						// Driving time and distances

						double divTimeA = driving_times.getDistance(k2, L); // Ir del ps1 al ps2

						// Update total time

						double totTimeA = walkTimeA + divTimeA + parkTime + servTime;

						double totalCost = calculateTotalCost(k2, L, w, v, m, j, tspTour);

						pH.addArc(l + "," + m, p + "," + j, totalCost, totTimeA,
								walkTimeA);
						PulseNode nodeI = pH.getNodes().get(l + "," + m);
						PulseNode nodeF = pH.getNodes().get(p + "," + j);
						if (totTimeA + nodeI.getMinTime()[1] < nodeF.getMinTime()[1]) {
							nodeF.getMinTime()[0] = totalCost + nodeI.getMinTime()[0];
							nodeF.getMinTime()[1] = totTimeA + nodeI.getMinTime()[1];
							nodeF.getMinTime()[2] = walkTimeA + nodeI.getMinTime()[2];
						}
						if (totalCost + nodeI.getMinCost()[0] < nodeF.getMinCost()[0]) {
							nodeF.getMinCost()[0] = totalCost + nodeI.getMinCost()[0];
							nodeF.getMinCost()[1] = totTimeA + nodeI.getMinCost()[1];
							nodeF.getMinCost()[2] = walkTimeA + nodeI.getMinCost()[2];
						}

						if (walkTimeA + nodeI.getMinDist()[2] < nodeF.getMinDist()[2]) {
							nodeF.getMinDist()[0] = totalCost + nodeI.getMinDist()[0];
							nodeF.getMinDist()[1] = totTimeA + nodeI.getMinDist()[1];
							nodeF.getMinDist()[2] = walkTimeA + nodeI.getMinDist()[2];
						}

					}
				}
			}
		}
	}

	/**
	 * This method adds only the new arcs !! This is excelent
	 * 
	 * @param l    vertical coordinate - parking spot
	 * @param m    horizontal coordinate - client
	 * @param last maximum horizontal coordinate
	 * @param beg  initial vertical coordinate
	 */
	public void develop2(int l, int m, int last, int beg, PulseHandler pH, TSPSolution tspTour) {

		// Initialize:

		double walkTime = 0;
		double servTime = 0;

		int w = tspTour.get(m); // Client!
		int u = w;

		// Main loop:

		for (int j = m + 1; j < last; j++) {
			int v = tspTour.get(j - 1); // Client!
			walkTime += walking_times.getDistance(u, v); // Going forward (between clients)
			servTime += data.getService_times().get(v);
			u = v;
		}

		for (int j = last; j <= last; j++) {
			int v = tspTour.get(j - 1); // Client!

			walkTime += walking_times.getDistance(u, v);// Going forward (between clients)
			servTime += data.getService_times().get(v);

			u = v;

			for (int p = beg; p <= last; p++) {
				if (p <= j) {

					// Update locations:

					int L = tspTour.get(p - 1); // Parking spot!

					int k2 = tspTour.get(l - 1); // Parking spot!

					// Update walking time

					double walkTimeA = walkTime + walking_times.getDistance(L, w) + walking_times.getDistance(v, L); // Movimiento
					// de
					// ps2
					// a
					// un
					// cliente
					// +
					// De
					// vuelta
					// desde
					// el
					// last
					// client
					// al
					// ps2
					double tourService = servTime;
					if (l == m && l != p) {
						tourService -= data.getService_times().get(tspTour.get(m - 1));
					}

					// Check if we must add the arc (This should go first)

					if (walkTimeA + tourService <= GlobalParameters.SUBTOUR_TIME_LIMIT
							&& walkTimeA <= GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT) {

						if ((l == p) || (p == j)) {

							// Parking time:

							double parkTime = 0;
							if (l != p) {
								parkTime = GlobalParameters.PARKING_TIME_MIN;
							}

							// Driving time and distances

							double divTimeA = driving_times.getDistance(k2, L); // Ir del ps1 al ps2

							// Coût de conduite standard

							// Update total time

							double totTimeA = walkTimeA + divTimeA + parkTime + servTime;

							// Coût total

							double totalCost = calculateTotalCost(k2, L, w, v, m, j, tspTour);

							pH.addArc(l + "," + m, p + "," + j, totalCost, totTimeA, walkTimeA);

							PulseNode nodeI = pH.getNodes().get(l + "," + m);
							PulseNode nodeF = pH.getNodes().get(p + "," + j);
							if (totTimeA + nodeI.getMinTime()[1] < nodeF.getMinTime()[1]) {
								nodeF.getMinTime()[0] = totalCost
										+ nodeI.getMinTime()[0];
								nodeF.getMinTime()[1] = totTimeA + nodeI.getMinTime()[1];
								nodeF.getMinTime()[2] = walkTimeA + nodeI.getMinTime()[2];
							}
							if (totalCost
									+ nodeI.getMinCost()[0] < nodeF.getMinCost()[0]) {
								nodeF.getMinCost()[0] = totalCost
										+ nodeI.getMinCost()[0];
								nodeF.getMinCost()[1] = totTimeA + nodeI.getMinCost()[1];
								nodeF.getMinCost()[2] = walkTimeA + nodeI.getMinCost()[2];
							}

							if (walkTimeA + nodeI.getMinDist()[2] < nodeF.getMinDist()[2]) {
								nodeF.getMinDist()[0] = totalCost
										+ nodeI.getMinDist()[0];
								nodeF.getMinDist()[1] = totTimeA + nodeI.getMinDist()[1];
								nodeF.getMinDist()[2] = walkTimeA + nodeI.getMinDist()[2];
							}
						}
					}

				}
			}
		}
	}

	/**
	 * This method adds only the new arcs !! This is excellent
	 * 
	 * @param i    vertical coordinate - parking spot
	 * @param k    horizontal coordinate - client
	 * @param last maximum horizontal coordinate
	 * @param beg  initial vertical coordinate
	 */
	public void developFirst2(int l, int m, int last, int beg, PulseHandler pH, TSPSolution tspTour) {

		// Initialize:

		double walkTime = 0;
		double servTime = 0;

		servTime += data.getService_times().get(tspTour.get(m - 1));

		int w = tspTour.get(m); // Client!
		int u = w;

		// Main loop:

		for (int j = m + 1; j < last; j++) {
			int v = tspTour.get(j - 1); // Client!
			walkTime += walking_times.getDistance(u, v); // Going forward (between clients)
			servTime += data.getService_times().get(v);
			u = v;
		}

		for (int j = last; j <= last; j++) {
			int v = tspTour.get(j - 1); // Client!

			walkTime += walking_times.getDistance(u, v); // Going forward (between clients)
			servTime += data.getService_times().get(v);

			u = v;

			for (int p = beg; p <= last; p++) {
				if (p <= j) {

					// Update locations:

					int L = tspTour.get(p - 1); // Parking spot!

					int k2 = tspTour.get(l - 1); // Parking spot!

					// Update walking time

					double walkTimeA = walkTime + walking_times.getDistance(L, w) + walking_times.getDistance(v, L); // Movimiento
																														// de
																														// ps2
																														// a
																														// un
																														// cliente
																														// +
																														// De
																														// vuelta
																														// desde
																														// el
																														// last
																														// client
																														// al
																														// ps2
					double tourService = servTime;
					if (l == m && l != p) {
						tourService -= data.getService_times().get(tspTour.get(m - 1));
					}

					// Check if we must add the arc (This should go first)

					if (walkTimeA + tourService <= GlobalParameters.SUBTOUR_TIME_LIMIT
							&& walkTimeA <= GlobalParameters.ROUTE_WALKING_DISTANCE_LIMIT) {

						if ((l == p) || (p == j)) {

							// Parking time:

							double parkTime = 0;
							if (l != p) {
								parkTime = GlobalParameters.PARKING_TIME_MIN;
							}

							// Driving time and distances

							double divTimeA = driving_times.getDistance(k2, L); // Ir del ps1 al ps2

							double totalCost = calculateTotalCost(k2, L, w, v, m, j, tspTour);

							// Update total time

							double totTimeA = walkTimeA + divTimeA + parkTime + servTime;

							pH.addArc(l + "," + m, p + "," + j, totalCost, totTimeA,
									walkTimeA);
							PulseNode nodeI = pH.getNodes().get(l + "," + m);
							PulseNode nodeF = pH.getNodes().get(p + "," + j);
							if (totTimeA + nodeI.getMinTime()[1] < nodeF.getMinTime()[1]) {
								nodeF.getMinTime()[0] = totalCost + nodeI.getMinTime()[0];
								nodeF.getMinTime()[1] = totTimeA + nodeI.getMinTime()[1];
								nodeF.getMinTime()[2] = walkTimeA + nodeI.getMinTime()[2];
							}
							if (totalCost + nodeI.getMinCost()[0] < nodeF.getMinCost()[0]) {
								nodeF.getMinCost()[0] = totalCost + nodeI.getMinCost()[0];
								nodeF.getMinCost()[1] = totTimeA + nodeI.getMinCost()[1];
								nodeF.getMinCost()[2] = walkTimeA + nodeI.getMinCost()[2];
							}

							if (walkTimeA + nodeI.getMinDist()[2] < nodeF.getMinDist()[2]) {
								nodeF.getMinDist()[0] = totalCost + nodeI.getMinDist()[0];
								nodeF.getMinDist()[1] = totTimeA + nodeI.getMinDist()[1];
								nodeF.getMinDist()[2] = walkTimeA + nodeI.getMinDist()[2];
							}
						}
					}

				}
			}
		}
	}

	/**
	 * This method recovers the route
	 * 
	 * @param route
	 * @return
	 */
	public String recoverRoute(ArrayList<String> route, TSPSolution tspTour, String ID) {
		String rta = "";
		rta += "CD -> ";
		for (int i = route.size() - 1; i >= 3; i--) {
			String[] nodeId1 = route.get(i).split(",");
			String[] nodeId2 = route.get(i - 1).split(",");
			int i1 = Integer.parseInt(nodeId1[0]);
			int j1 = Integer.parseInt(nodeId1[1]);
			int i2 = Integer.parseInt(nodeId2[0]);
			int j2 = Integer.parseInt(nodeId2[1]);
			rta += calculateRouteAux(i1 - 1, j1 - 1, i2 - 1, j2 - 1, tspTour) + " || ";
		}

		String[] nodeId1 = route.get(2).split(",");
		String[] nodeId2 = route.get(1).split(",");
		int i1 = Integer.parseInt(nodeId1[0]);
		int j1 = Integer.parseInt(nodeId1[1]);
		int i2 = Integer.parseInt(nodeId2[0]);
		int j2 = Integer.parseInt(nodeId2[1]);
		rta += calculateRouteAux(i1 - 1, j1 - 1, i2 - 1, j2 - 1, tspTour) + " -> CD ";
		return rta;
	}

	/**
	 * 
	 * @param i1
	 * @param j1
	 * @param i2
	 * @param j2
	 * @return
	 */
	public String calculateRouteAux(int i1, int j1, int i2, int j2, TSPSolution tspTour) {
		String route = "";
		if (i1 == i2) {
			route += tspTour.get(i1) + " --- ";
			for (int j = j1 + 1; j <= j2; j++) {
				if (j > i1) {
					route += tspTour.get(j) + " --- ";
				}
			}
			route += tspTour.get(i2);
		} else if (j2 == i2 && j1 + 1 == j2) {
			route += tspTour.get(i1) + " -> " + tspTour.get(i2);
		} else {
			route += tspTour.get(i1) + " -> " + tspTour.get(i2);
			for (int j = j1 + 1; j <= j2; j++) {
				if (j != i2 || (j > j1 + 1 && j < j2)) {
					route += " --- " + tspTour.get(j);
				}
			}
			route += " --- " + tspTour.get(i2);
		}
		return route;
	}

	/**
	 * Calculates the total cost for a route segment, including custom costs if
	 * available.
	 *
	 * @param k2      The starting parking spot index.
	 * @param L       The ending parking spot index.
	 * @param w       The starting client index.
	 * @param v       The ending client index.
	 * @param m       The starting index for the TSP tour segment. w =
	 *                tspTour.get(m)
	 * @param j       The ending index for the TSP tour segment. v =
	 *                tspTour.get(j - 1)
	 * @param tspTour The TSP tour containing the sequence of clients.
	 * @return The total cost for the route segment.
	 */
	public double calculateTotalCost(int k2, int L, int w, int v, int m, int j, TSPSolution tspTour) {
		// Standard driving cost
		double divDisA = distances.getDistance(k2, L);
		double divCostA = divDisA * GlobalParameters.VARIABLE_COST;

		// assert that w and v correspond to m and j in the tspTour
		if (tspTour.get(m) != w || tspTour.get(j - 1) != v) {
			throw new IllegalArgumentException("TSP tour indices do not match client indices.");
		}

		// Check for custom driving cost
		if (customArcCosts.hasCustomCost(k2, L, 1)) {
			divCostA = customArcCosts.getCustomCost(k2, L, 1);
		}

		// Coût de marche total
		double totalWalkingCost = 0;

		// 1. Coût de marche du parking au premier client
		if (customArcCosts.hasCustomCost(L, w, 2)) {
			totalWalkingCost += customArcCosts.getCustomCost(L, w, 2);
		}
		// 2. Coûts de marche entre clients
		int prevClient = w;
		for (int idx = m + 1; idx < j; idx++) {
			int nextClient = tspTour.get(idx);
			if (customArcCosts.hasCustomCost(prevClient, nextClient, 2)) {
				totalWalkingCost += customArcCosts.getCustomCost(prevClient, nextClient, 2);
			}
			prevClient = nextClient;
		}

		// 3. Coût de marche du dernier client au parking
		if (customArcCosts.hasCustomCost(v, L, 2)) {
			totalWalkingCost += customArcCosts.getCustomCost(v, L, 2);
		}

		// Total
		return divCostA + totalWalkingCost;
	}

}
