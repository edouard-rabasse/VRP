package util;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class DuplicateFile {
    public static void DuplicateFile(String sourcePath, String destinationPath) {
        try (FileInputStream in = new FileInputStream(sourcePath);
                FileOutputStream out = new FileOutputStream(destinationPath)) {

            byte[] buffer = new byte[1024]; // 1 Ko
            int length;

            while ((length = in.read(buffer)) > 0) {
                out.write(buffer, 0, length);
            }

            System.out.println("Fichier dupliqué avec succès !");
        } catch (IOException e) {
            System.err.println("Erreur lors de la duplication : " + e.getMessage());
        }
    }
}