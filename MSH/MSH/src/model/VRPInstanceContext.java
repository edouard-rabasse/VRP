package model;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import distanceMatrices.DepotToCustomersDistanceMatrixV2;
import distanceMatrices.ArcModificationMatrix;
import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;

/**
 * Encapsulates all the data structures and matrices needed for a VRP instance.
 * This class provides a clean interface for initializing and accessing
 * instance-specific data like distance matrices, driving times, and walking
 * times.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class VRPInstanceContext {

    private final DataHandler data;
    private final ArrayDistanceMatrix distances;
    private final ArrayDistanceMatrix drivingTimes;
    private final ArrayDistanceMatrix walkingTimes;
    private final String instanceIdentifier;
    private final String instanceName;

    /**
     * Creates a VRP instance context with standard distance matrices
     */
    public VRPInstanceContext(String instanceIdentifier) throws IOException {
        this.instanceIdentifier = instanceIdentifier;
        this.instanceName = extractInstanceName(instanceIdentifier);

        // Initialize data handler
        this.data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instanceIdentifier);

        // Initialize distance matrices
        this.distances = new DepotToCustomersDistanceMatrix(data);
        this.drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        this.walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);
    }

    /**
     * Creates a VRP instance context with modified distance matrices for fixed
     * edges
     */
    public VRPInstanceContext(String instanceIdentifier, ArcModificationMatrix arcModificationMatrix)
            throws IOException {
        this.instanceIdentifier = instanceIdentifier;
        this.instanceName = extractInstanceName(instanceIdentifier);

        // Initialize data handler
        this.data = new DataHandler(GlobalParameters.COORDINATES_FOLDER + instanceIdentifier);

        // Initialize distance matrices with modifications
        // Use modified distances incorporating arc modifications
        this.distances = new DepotToCustomersDistanceMatrixV2(data, arcModificationMatrix);

        this.drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        this.walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);
    }

    /**
     * Creates a VRP instance context with custom cost handling
     */
    public VRPInstanceContext(String instanceIdentifier, CustomArcCostMatrix customCosts) throws IOException {
        this.instanceIdentifier = instanceIdentifier;
        this.instanceName = extractInstanceName(instanceIdentifier);

        // Initialize data handler
        this.data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instanceIdentifier);

        // Initialize distance matrices
        this.distances = new DepotToCustomersDistanceMatrix(data);
        this.drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        this.walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);
    }

    private String extractInstanceName(String instanceIdentifier) {
        return instanceIdentifier.replace(".txt", "").replace("Coordinates_", "");
    }

    // Getters
    public DataHandler getData() {
        return data;
    }

    public ArrayDistanceMatrix getDistances() {
        return distances;
    }

    public ArrayDistanceMatrix getDrivingTimes() {
        return drivingTimes;
    }

    public ArrayDistanceMatrix getWalkingTimes() {
        return walkingTimes;
    }

    public String getInstanceIdentifier() {
        return instanceIdentifier;
    }

    public String getInstanceName() {
        return instanceName;
    }
}
