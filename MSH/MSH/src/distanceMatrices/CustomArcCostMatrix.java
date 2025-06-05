package distanceMatrices;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Cette classe permet de stocker les coûts personnalisés pour certains arcs
 */
public class CustomArcCostMatrix {

    // Map qui stocke les coûts personnalisés - clé: "tail;head;mode", valeur: coût
    private Map<String, Double> customCosts;

    /**
     * Constructeur
     */
    public CustomArcCostMatrix() {
        this.customCosts = new HashMap<>();
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
}