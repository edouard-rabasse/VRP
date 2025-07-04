package msh;

import java.io.IOException;
import java.util.ArrayList;

import core.ArrayDistanceMatrix;
import core.RoutePool;
import core.Split;
import dataStructures.DataHandler;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import globalParameters.GlobalParameters;
import split.SplitPLRP;

public class MSHContext {
    public DataHandler data;
    public ArrayDistanceMatrix distances;
    public ArrayDistanceMatrix drivingTimes;
    public ArrayDistanceMatrix walkingTimes;
    public ArrayList<RoutePool> pools;
    public AssemblyFunction assembler;
    public MSH msh;
    public Split split; // Not public as it may be replaced with custom implementations
    public int numIterations;

    public MSHContext(DataHandler data, ArrayDistanceMatrix distances, ArrayDistanceMatrix drivingTimes,
            ArrayDistanceMatrix walkingTimes, ArrayList<RoutePool> pools, AssemblyFunction assembler,
            MSH msh, Split split, int numIterations) {
        this.data = data;
        this.distances = distances;
        this.drivingTimes = drivingTimes;
        this.walkingTimes = walkingTimes;
        this.pools = pools;
        this.assembler = assembler;
        this.msh = msh;
        this.split = split;
        this.numIterations = numIterations;
    }

    public static MSHContext initializeMSH(String instance_identifier) throws IOException {

        // Read the instance
        DataHandler data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

        // Create distance matrices
        ArrayDistanceMatrix distances = new DepotToCustomersDistanceMatrix(data);
        ArrayDistanceMatrix drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        ArrayDistanceMatrix walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);

        // Initialize pools and assembler
        ArrayList<RoutePool> pools = new ArrayList<RoutePool>();
        AssemblyFunction assembler = new GurobiSetPartitioningSolver(data.getNbCustomers(), true, data);
        MSH msh = new MSH(assembler, GlobalParameters.THREADS);

        // Initialize split algorithm
        Split split = new SplitPLRP(distances, drivingTimes, walkingTimes, data);

        // Calculate iterations
        int numIterations = Math.max(1, (int) Math.ceil(GlobalParameters.MSH_NUM_ITERATIONS / 8.0));

        return new MSHContext(data, distances, drivingTimes, walkingTimes, pools, assembler, msh, split, numIterations);
    }

}
