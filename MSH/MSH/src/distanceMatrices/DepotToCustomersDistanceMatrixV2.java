package distanceMatrices;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import util.EuclideanCalculator;
import distanceMatrices.ArcModificationMatrix;

/**
 * This class implements an instance of a distance matrix, for the PLRP
 * instances
 * 
 * It holds the distances in kilometers between two nodes
 * 
 * @author nicolas.cabrera-malik
 */
public class DepotToCustomersDistanceMatrixV2 extends ArrayDistanceMatrix {

	/**
	 * Constructs the distance matrix
	 * 
	 * @throws IOException
	 */

	public DepotToCustomersDistanceMatrixV2(DataHandler data, ArcModificationMatrix arcModificationMatrix)
			throws IOException {

		super();

		// Number of nodes:

		int dimension = data.getX_coors().size();

		// Initializes the distance matrix:

		double[][] distances = new double[dimension][dimension];

		// Fills the matrix:

		// Between customers:

		EuclideanCalculator euc = new EuclideanCalculator();
		for (int i = 0; i < dimension; i++) {

			for (int j = 0; j < dimension; j++) {
				if (arcModificationMatrix.get(i, j)) {
					// If the arc is modified, we calculate the distance
					distances[i][j] = euc.calc(data.getX_coors().get(i), data.getY_coors().get(i),
							data.getX_coors().get(j),
							data.getY_coors().get(j));
				} else {

					System.out.println("Null distance between " + i + " and " + j);
					distances[i][j] = GlobalParameters.FIXED_ARCS_DISTANCE;
				}

			}

		}

		// Sets the distance matrix:

		this.setDistances(distances);
	}
}
