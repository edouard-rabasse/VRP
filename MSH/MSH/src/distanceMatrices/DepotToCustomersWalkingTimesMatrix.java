package distanceMatrices;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;
import globalParameters.GlobalParameters;
import util.EuclideanCalculator;

/**
 * This class implements an instance of a distance matrix, for the PLRP instances
 * 
 * It holds the walking times in minutes between two nodes
 * 
 * @author nicolas.cabrera-malik
 */
public class DepotToCustomersWalkingTimesMatrix extends ArrayDistanceMatrix{

	/**
	 * Constructs the distance matrix
	 * @throws IOException 
	 */
	
	public DepotToCustomersWalkingTimesMatrix(DataHandler data) throws IOException {
		
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
						
					
					double dist = euc.calc(data.getX_coors().get(i), data.getY_coors().get(i), data.getX_coors().get(j), data.getY_coors().get(j)) ;
					if(dist <= GlobalParameters.MAX_WD_B2P) {
						distances[i][j] = dist * (1.0/GlobalParameters.WALKING_SPEED) * 60;
					}else {
						distances[i][j] = Double.POSITIVE_INFINITY;	
					}
				}
					
			}

		// Sets the distance matrix:
		
		this.setDistances(distances);
	}
}
