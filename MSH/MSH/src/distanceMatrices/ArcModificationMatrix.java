package distanceMatrices;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ArcModificationMatrix {

    // Map of (head, tail) -> to_modify (true/false)
    private final Map<String, Boolean> matrix = new HashMap<>();

    /**
     * Loads the matrix from the given file path.
     * Each line is formatted as: head;tail;mode;route;to_modify
     */
    public void loadFromFile(String filePath) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(";");
            if (parts.length < 5)
                continue;

            int head = Integer.parseInt(parts[0]);
            int tail = Integer.parseInt(parts[1]);
            boolean toModify = parts[4].trim().equals("1");

            matrix.put(getKey(head, tail), toModify);
        }

        reader.close();
    }

    /**
     * Returns true if the arc (head, tail) is marked to be modified.
     */
    public boolean get(int head, int tail) {
        return matrix.getOrDefault(getKey(head, tail), true);
    }

    /**
     * Utility method to create a unique key for (head, tail)
     */
    private String getKey(int head, int tail) {
        return head + "_" + tail;
    }
}
