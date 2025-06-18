package model;

import java.io.IOException;
import java.io.File;

import distanceMatrices.CustomArcCostMatrix;
import globalParameters.GlobalParameters;
import dataStructures.DataHandler;
import core.ArrayDistanceMatrix;

/**
 * Manages custom arc cost operations for VRP instances.
 * This class handles the setup, loading, and saving of custom arc costs,
 * including penalty applications for flagged arcs.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class CustomArcCostManager {

    private final CustomArcCostMatrix customArcCosts;

    public CustomArcCostManager(CustomArcCostMatrix customArcCosts) {
        this.customArcCosts = customArcCosts;
    }

    /**
     * Sets up custom costs for the given instance and parameters
     */
    public CustomArcCostMatrix setupCustomCosts(String instanceIdentifier, String costFile,
            String arcPath, int suffix, DataHandler data,
            ArrayDistanceMatrix distances) throws IOException {

        String instanceName = instanceIdentifier.replace(".txt", "").replace("Coordinates_", "");
        int depot = data.getNbCustomers() + 1;

        // Initialize custom arc cost matrix
        CustomArcCostMatrix arcCost = new CustomArcCostMatrix();
        arcCost.addDepot(depot);

        // Load existing custom costs
        arcCost.loadFromFile(GlobalParameters.ARCS_MODIFIED_FOLDER + costFile);

        // Update costs based on flagged arcs
        String globalArcPath = GlobalParameters.RESULT_FOLDER + arcPath;
        if (new File(globalArcPath).isFile()) {
            arcCost.updateFromFlaggedFile(globalArcPath,
                    GlobalParameters.CUSTOM_COST_MULTIPLIER, distances,
                    GlobalParameters.DEFAULT_WALK_COST);
        }

        // Save updated costs
        arcCost.saveFile(GlobalParameters.ARCS_MODIFIED_FOLDER + "Costs_" + instanceName + "_" + (suffix + 1) + ".txt");

        return arcCost;
    }

    public CustomArcCostMatrix getCustomArcCosts() {
        return customArcCosts;
    }
}
