package globalParameters;

/**
 * This class contains the main parameters for the MSH algorithm
 * 
 * @author nicolas.cabrera-malik
 *
 */
public class GlobalParameters {

	// Relative paths:
	
		/**
		 * Instance folder where the instances are stored
		 */
		public static final String INSTANCE_FOLDER = GlobalParametersReader.<String>get("INSTANCE_FOLDER", String.class);
		
		/**
		 * Folder in which all the results will be printed
		 */
		public static final String RESULT_FOLDER = GlobalParametersReader.<String>get("RESULT_FOLDER", String.class);
		
	// Precision:
		
		public static final int PRECISION = GlobalParametersReader.<Integer>get("PRECISION", Integer.class);
		public static final double DECIMAL_PRECISION = Math.pow(10, -PRECISION);

	// Experiment parameters:

		/**
		 * Seed for the current run (to allow for replication)
		 */
		public static int SEED = GlobalParametersReader.<Integer>get("SEED", Integer.class);
		
		/**
		 * Number of threads for the sampling phase. How many sampling functions will work in parallel.
		 */
		public static int THREADS = GlobalParametersReader.<Integer>get("THREADS", Integer.class);
		
		/**
		 * Should we print in console some information while we run the algorithm?
		 */
		public static final boolean PRINT_IN_CONSOLE = GlobalParametersReader.<String>get("PRINT_IN_CONSOLE", String.class).equals("false") ? false:true;
		
		/**
		 * Should we print in a txt file the pools?
		 */
		public static final boolean PRINT_POOLS_TO_FILE = GlobalParametersReader.<String>get("PRINT_POOLS_TO_FILE", String.class).equals("false") ? false:true;
		
	// Main constraints:
		
		/**
		 * Driving speed in km/h
		 */

		public static int DRIVING_SPEED = GlobalParametersReader.<Integer>get("DRIVING_SPEED", Integer.class);
		
		/**
		 * Walking speed in km/h
		 */

		public static int WALKING_SPEED = GlobalParametersReader.<Integer>get("WALKING_SPEED", Integer.class);
		
		/**
		 * Fixed cost
		 */

		public static int FIXED_COST = GlobalParametersReader.<Integer>get("FIXED_COST", Integer.class);
		
		/**
		 * Variable cost
		 */

		public static int VARIABLE_COST = GlobalParametersReader.<Integer>get("VARIABLE_COST", Integer.class);
		
		/**
		 * Max walking distance between two points
		 */

		public static double MAX_WD_B2P = GlobalParametersReader.<Double>get("MAX_WD_B2P", Double.class);
		
		/**
		 * Time limit for each subtour
		 */

		public static int SUBTOUR_TIME_LIMIT = GlobalParametersReader.<Integer>get("SUBTOUR_TIME_LIMIT", Integer.class);
		
		/**
		 * Walking distance limit for each route
		 */

		public static double ROUTE_WALKING_DISTANCE_LIMIT = GlobalParametersReader.<Double>get("ROUTE_WALKING_DISTANCE_LIMIT", Double.class);
			
		/**
		 * Walking duration limit in minutes
		 */

		public static double ROUTE_DURATION_LIMIT = GlobalParametersReader.<Double>get("ROUTE_DURATION_LIMIT", Double.class);
		
		/**
		 * Parking time in minutes
		 */

		public static double PARKING_TIME_MIN = GlobalParametersReader.<Double>get("PARKING_TIME_MIN", Double.class);
		
		
	// MSH  parameters:
		
		/**
		 * Maximum number of routes allowed in each pool
		 */
		public static final int MSH_MAX_POOL_SIZE = GlobalParametersReader.<Integer>get("MSH_MAX_POOL_SIZE", Integer.class);
		
		/**
		 * Maximum number of iterations for the MSH
		 */
		public static final int MSH_NUM_ITERATIONS = GlobalParametersReader.<Integer>get("MSH_NUM_ITERATIONS", Integer.class);
		
		/**
		 * Time limit for the sampling phase in seconds
		 */
		public static final int MSH_SAMPLING_TIME_LIMIT = GlobalParametersReader.<Integer>get("MSH_SAMPLING_TIME_LIMIT", Integer.class);
		
		/**
		 * Randomization factor (large value) for the tsp heuristics
		 */
		public static final int MSH_RANDOM_FACTOR_HIGH = GlobalParametersReader.<Integer>get("MSH_RANDOM_FACTOR_HIGH", Integer.class);
		
		/**
		 * Randomization factor (large value) for the tsp heuristics
		 */
		public static final int MSH_RANDOM_FACTOR_HIGH_RN = GlobalParametersReader.<Integer>get("MSH_RANDOM_FACTOR_HIGH_RN", Integer.class);
		
		/**
		 * Randomization factor (small value) for the tsp heuristics
		 */
		public static final int MSH_RANDOM_FACTOR_LOW = GlobalParametersReader.<Integer>get("MSH_RANDOM_FACTOR_LOW", Integer.class);
		
		/**
		 * Time limit for the assembly phase in seconds
		 */
		public static final int MSH_ASSEMBLY_TIME_LIMIT = GlobalParametersReader.<Integer>get("MSH_ASSEMBLY_TIME_LIMIT", Integer.class);
		
}
