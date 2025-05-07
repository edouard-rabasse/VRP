package distanceMatrices;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;
import util.EuclideanCalculator;

/**
 * This class implements an instance of a distance matrix, for the PLRP instances
 * 
 * It holds the distances in kilometers between two nodes
 * 
 * @author nicolas.cabrera-malik
 */
public class DepotToCustomersDistanceMatrix extends ArrayDistanceMatrix{

	/**
	 * Constructs the distance matrix
	 * @throws IOException 
	 */
	
	public DepotToCustomersDistanceMatrix(DataHandler data) throws IOException {
		
		super();
		
		// Number of nodes:
		
		int dimension = data.getX_coors().size();
		
		// Initializes the distance matrix:
		
		double[][] distances = new double[dimension][dimension];
		
		// Fills the matrix:
		
			//Between customers:
		
			EuclideanCalculator euc = new EuclideanCalculator();
			for(int i = 0; i < dimension; i++) {
					
				for(int j = 0; j < dimension; j++) {
						
					distances[i][j] = euc.calc(data.getX_coors().get(i), data.getY_coors().get(i), data.getX_coors().get(j), data.getY_coors().get(j));
						
				}
					
			}

		// Sets the distance matrix:
		
		this.setDistances(distances);
	}
}
