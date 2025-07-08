package distanceMatrices;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import core.ArrayDistanceMatrix;
import dataStructures.DataHandler;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import globalParameters.GlobalParameters;

/**
 * CustomArcCostMatrix is a class for managing custom arc costs in a network,
 * supporting multiple transportation modes (e.g., driving, walking).
 * 
 * <p>
 * It allows adding, retrieving, and updating custom costs for arcs between
 * nodes,
 * with special handling for depot nodes. Costs can be loaded from or saved to
 * files,
 * and updated based on flagged arcs (e.g., for penalizing certain routes).
 * </p>
 * 
 * <p>
 * Key features:
 * <ul>
 * <li>Store and retrieve custom costs for arcs, indexed by tail, head, and
 * mode.</li>
 * <li>Support for loading and saving custom costs from/to files.</li>
 * <li>Update costs based on flagged arcs, with support for cost
 * multipliers.</li>
 * <li>Special handling for depot arcs.</li>
 * </ul>
 * </p>
 * 
 * <p>
 * Usage example:
 * 
 * <pre>
 * CustomArcCostMatrix matrix = new CustomArcCostMatrix();
 * matrix.addDepot(0);
 * matrix.addCustomCost(1, 2, 1, 10.0);
 * double cost = matrix.getCustomCost(1, 2, 1);
 * </pre>
 * </p>
 * 
 * <p>
 * File formats:
 * <ul>
 * <li>Custom cost file: <code>tail;head;mode;cost</code></li>
 * <li>Flagged file:
 * <code>tail;head;mode;route number;flagged (1 or 0)</code></li>
 * </ul>
 * </p>
 * 
 * <p>
 * Thread safety: This class is not thread-safe.
 * </p>
 * 
 * @author edouard-rabasse
 * @version 1.0
 */
public class CustomArcCostMatrix {

    // Map qui stocke les coûts personnalisés - clé: "tail;head;mode", valeur: coût
    private Map<String, Double> customCosts;

    private int depot;

    /**
     * Constructor
     */
    public CustomArcCostMatrix() {
        this.customCosts = new HashMap<>();

    }

    public void addDepot(int depot) {
        this.depot = depot;
    }

    /**
     * Ajoute un coût personnalisé pour un arc
     * 
     * @param tail Nœud de départ
     * @param head Nœud d'arrivée
     * @param mode Mode (1=voiture, 2=piéton)
     * @param cost Coût personnalisé
     */
    public void addCustomCost(int tail, int head, int mode, double cost) {
        if (tail == depot) {
            String key = head + ";" + 0 + ";" + mode;
            customCosts.put(key, cost);
        }
        if (head == depot) {
            String key = 0 + ";" + tail + ";" + mode;
            customCosts.put(key, cost);
        }
        String key = tail + ";" + head + ";" + mode;
        customCosts.put(key, cost);
    }

    /**
     * Vérifie si un arc a un coût personnalisé
     * 
     * @param tail Nœud de départ
     * @param head Nœud d'arrivée
     * @param mode Mode (1=voiture, 2=piéton)
     * @return true si l'arc a un coût personnalisé
     */
    public boolean hasCustomCost(int tail, int head, int mode) {
        String key = tail + ";" + head + ";" + mode;
        return customCosts.containsKey(key);
    }

    /**
     * Retourne le coût personnalisé pour un arc
     * 
     * @param tail Nœud de départ
     * @param head Nœud d'arrivée
     * @param mode Mode (1=voiture, 2=piéton)
     * @return Le coût personnalisé ou -1 si non défini
     */
    public double getCustomCost(int tail, int head, int mode) {
        String key = tail + ";" + head + ";" + mode;
        return customCosts.getOrDefault(key, -1.0);
    }

    /**
     * Charge les coûts personnalisés depuis un fichier
     * Format: tail;head;mode;cost
     * 
     * @param filePath Chemin du fichier
     * @throws IOException
     */
    public void loadFromFile(String filePath) throws IOException {
        if (!new File(filePath).isFile()) {
            if (GlobalParameters.PRINT_IN_CONSOLE) {
                System.out.println(
                        "[CustomCost creation] No custom cost file " + filePath + " provided, skipping update.");
            }
            return;
        }
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split(";");
            if (parts.length >= 4) {
                int tail = Integer.parseInt(parts[0]);
                int head = Integer.parseInt(parts[1]);
                int mode = Integer.parseInt(parts[2]);
                double cost = Double.parseDouble(parts[3]);

                addCustomCost(tail, head, mode, cost);
            }
        }

        reader.close();
    }

    public void printCustomCosts() {
        for (Map.Entry<String, Double> entry : customCosts.entrySet()) {
            System.out.println("Arc: " + entry.getKey() + " - Cost: " + entry.getValue());
        }
    }

    /**
     * Saves the custom costs to a file
     * Format: tail;head;mode;cost
     * 
     * @param filePath
     */
    public void saveFile(String filePath) {
        try {
            File file = new File(filePath);
            File parentDir = file.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                for (Map.Entry<String, Double> entry : customCosts.entrySet()) {
                    String[] parts = entry.getKey().split(";");
                    int tail = Integer.parseInt(parts[0]);
                    int head = Integer.parseInt(parts[1]);
                    int mode = Integer.parseInt(parts[2]);
                    double cost = entry.getValue();
                    writer.write(tail + ";" + head + ";" + mode + ";" + cost);
                    writer.newLine();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * Updates the custom costs from a flagged file
     * 
     * @param filePath  Path to the flagged file. Each line is expected to be in the
     *                  format:
     *                  "tail;head;mode;route number; flagged (1 or 0)"
     * @param lambda    Multiplier factor for flagged arcs (cost *= 1 + lambda)
     * @param distances Distance matrix for calculating default driving costs
     * @param alpha     Default cost for walking arcs (mode 2)
     * @throws IOException
     */
    public void updateFromFlaggedFile(String filePath, double lambda,
            ArrayDistanceMatrix distances, double alpha) throws IOException {

        // check if the filePath corresponds to a valid file

        if (!new File(filePath).isFile()) {
            if (GlobalParameters.PRINT_IN_CONSOLE) {
                System.out.println("[CustomCost update] No flagged file " + filePath + " provided, skipping update.");
            }
            return;

        }
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println("[CustomCost update] Starting to update custom costs from flagged file: " + filePath);
        }

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty())
                continue;

            String[] parts = line.split(";");
            if (parts.length >= 5) {
                try {
                    int tail = Integer.parseInt(parts[0]);
                    int head = Integer.parseInt(parts[1]);
                    int mode = Integer.parseInt(parts[2]);
                    // parts[3] is route number - not used for cost calculation
                    int flagged = Integer.parseInt(parts[4]);

                    if (flagged == 1) {
                        String key = tail + ";" + head + ";" + mode;

                        if (hasCustomCost(tail, head, mode) && customCosts.get(key) > 0.0) {
                            // Arc already has positive custom cost - multiply by (1 + lambda)
                            double currentCost = getCustomCost(tail, head, mode);
                            double newCost = currentCost * (1.0 + lambda);
                            customCosts.put(key, newCost);

                            if (GlobalParameters.PRINT_IN_CONSOLE) {
                                // Log the update
                                System.out.println("[CustomCost] Updated arc " + tail + "->" + head +
                                        " (mode " + mode + ") from " + currentCost + " to " + newCost);
                            }
                        } else {
                            // Arc doesn't have custom cost yet - create default cost
                            double defaultCost;
                            if (mode == 1) {
                                // Driving mode - use distance * VARIABLE_COST
                                if (GlobalParameters.PRINT_IN_CONSOLE) {
                                    System.out.println("[CustomCost] Creating new arc " + tail + "->" + head +
                                            " (mode " + mode + ") with default cost based on distance.");
                                }
                                defaultCost = distances.getDistance(tail % this.depot, head % this.depot)
                                        * GlobalParameters.VARIABLE_COST;
                            } else {
                                // Walking mode - use alpha
                                defaultCost = alpha;
                            }

                            // Apply the lambda factor immediately since it's flagged
                            double newCost = defaultCost * (1.0 + lambda);
                            customCosts.put(key, newCost);

                        }
                    }
                } catch (NumberFormatException e) {
                    System.err.println("[CustomCost] Error parsing line: " + line + " - " + e.getMessage());
                }
            }
        }

        reader.close();
    }

    /**
     * Overloaded method with default parameters
     */
    public void updateFromFlaggedFile(String filePath) throws IOException {
        // You'll need to pass these parameters from the calling code
        throw new UnsupportedOperationException("Use updateFromFlaggedFile(filePath, lambda, distances, alpha)");
    }

    /**
     * Create constraint matrix for upper right corner
     */
    public static CustomArcCostMatrix createUpperRightConstraintMatrix(DataHandler data) {
        CustomArcCostMatrix constraintMatrix = new CustomArcCostMatrix();
        int depot = data.getNbCustomers() + 1;
        constraintMatrix.addDepot(depot);

        for (int i = 0; i < data.getNbCustomers(); i++) {
            double xCoord = data.getX_coors().get(i);
            double yCoord = data.getY_coors().get(i);

            if (xCoord > 5.0 && yCoord > 5.0) {
                for (int j = 0; j < data.getNbCustomers(); j++) {
                    if (i != j) {
                        constraintMatrix.addCustomCost(i, j, 2, GlobalParameters.FIXED_ARCS_DISTANCE * 1000);
                    }
                }

            }
        }

        return constraintMatrix;
    }

    /**
     * Transforme tous les coûts d'arcs en utilisant un mapping d'indices
     * Cette fonction peut être utilisée pour convertir une matrice globale en
     * locale ou vice versa
     * 
     * @param mapping         Map contenant la correspondance entre indices
     *                        (local→global ou global→local)
     * @param isLocalToGlobal true si mapping est local→global, false si
     *                        global→local
     * @return Une nouvelle matrice de coûts avec les indices convertis
     */
    public CustomArcCostMatrix applyMapping(Map<Integer, Integer> mapping, boolean isLocalToGlobal) {
        CustomArcCostMatrix mappedCosts = new CustomArcCostMatrix();

        // Créer un mapping inverse si nécessaire
        Map<Integer, Integer> inverseMapping = null;
        if (!isLocalToGlobal) {
            // Si on fournit un mapping global→local, on l'utilise directement
            inverseMapping = new HashMap<>(mapping);
        } else {
            // Si on fournit un mapping local→global, on crée l'inverse
            inverseMapping = new HashMap<>();
            for (Map.Entry<Integer, Integer> entry : mapping.entrySet()) {
                inverseMapping.put(entry.getValue(), entry.getKey());
            }
        }

        // Traiter le dépôt spécialement
        inverseMapping.put(0, 0);

        System.out.println("Converting costs with mapping: " + (isLocalToGlobal ? "local→global" : "global→local"));
        System.out.println("Mapping: " + mapping);

        // Parcourir tous les coûts et les convertir
        int countConverted = 0;
        for (Map.Entry<String, Double> entry : customCosts.entrySet()) {
            String key = entry.getKey();
            Double cost = entry.getValue();

            // Décomposer la clé pour obtenir tail, head, mode
            String[] parts = key.split(";");
            int tail = Integer.parseInt(parts[0]);
            int head = Integer.parseInt(parts[1]);
            int mode = Integer.parseInt(parts[2]);

            // Convertir les indices
            Integer mappedTail = getMappedIndex(tail, inverseMapping);
            Integer mappedHead = getMappedIndex(head, inverseMapping);

            // Ajouter le coût converti s'il est valide
            if (mappedTail != null && mappedHead != null) {
                mappedCosts.addCustomCost(mappedTail, mappedHead, mode, cost);
                countConverted++;

                if (GlobalParameters.PRINT_IN_CONSOLE) {
                    System.out.println("Converted cost: (" + tail + "," + head + "," + mode + ") → (" +
                            mappedTail + "," + mappedHead + "," + mode + ") = " + cost);
                }
            }
        }

        System.out.println("Converted " + countConverted + " costs out of " + customCosts.size());
        return mappedCosts;
    }

    /**
     * Obtient l'indice converti en utilisant le mapping
     * 
     * @param index   Indice à convertir
     * @param mapping Mapping à utiliser
     * @return Indice converti ou null si non trouvé
     */
    private Integer getMappedIndex(int index, Map<Integer, Integer> mapping) {
        // Le dépôt reste le dépôt
        if (index == 0) {
            return 0;
        }

        // Utiliser le mapping pour les autres indices
        return mapping.getOrDefault(index, null);
    }

}