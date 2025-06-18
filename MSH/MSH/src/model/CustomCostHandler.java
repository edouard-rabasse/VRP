package model;

import java.io.IOException;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;
import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;

/**
 * Handles custom cost matrix operations for VRP instances.
 * This class encapsulates the logic for loading, updating, and saving custom
 * arc costs.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class CustomCostHandler {

    private final CustomArcCostMatrix customCosts;
    private final DataHandler data;

    public CustomCostHandler(DataHandler data) {
        this.data = data;
        this.customCosts = new CustomArcCostMatrix();

        // Add depot information
        int depot = data.getNbCustomers() + 1;
        customCosts.addDepot(depot);
    }

    /**
     * Loads custom costs from file and updates based on flagged arcs
     */
    public void loadAndUpdateCosts(String costFile, String arcPath, ArrayDistanceMatrix distances)
            throws IOException {
        // Load initial costs from file
        customCosts.loadFromFile(GlobalParameters.ARCS_MODIFIED_FOLDER + costFile);

        // Update costs based on flagged arcs
        customCosts.updateFromFlaggedFile(
                GlobalParameters.RESULT_FOLDER + arcPath,
                GlobalParameters.CUSTOM_COST_MULTIPLIER,
                distances,
                GlobalParameters.DEFAULT_WALK_COST);
    }

    /**
     * Saves the custom costs to a file
     */
    public void saveCosts(String instanceName, int suffix) throws IOException {
        String outputPath = GlobalParameters.ARCS_MODIFIED_FOLDER + "Costs_" + instanceName + "_" +
                (suffix) + ".txt";
        customCosts.saveFile(outputPath);
    }

    /**
     * Gets the custom cost matrix
     */
    public CustomArcCostMatrix getCustomCosts() {
        return customCosts;
    }
}
