package util;

import java.io.PrintWriter;

import core.Route;
import core.RouteAttribute;

public class RouteProcessor {
    /**
     * Process individual route for solution output
     */
    public static void processRoute(Route route, PrintWriter writer, int routeId, int nbCustomers) {
        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
        chain = chain.replace(" 0 ", " " + (nbCustomers + 1) + " ");
        chain = chain.replace("CD", "" + (nbCustomers + 1));
        chain = chain.replace(" ", "");
        String[] parts = chain.split("[||]");

        for (String part : parts) {
            if (part.length() > 0) {
                processRoutePart(part, writer, routeId);
            }
        }
    }

    /**
     * Process individual part of route chain
     */
    private static void processRoutePart(String part, PrintWriter writer, int routeId) {
        if (part.contains("->") && !part.contains("---")) {
            // Only driving
            processSimpleMode(part, "->", 1, writer, routeId);
        } else if (!part.contains("->") && part.contains("---")) {
            // Only walking
            processSimpleMode(part, "---", 2, writer, routeId);
        } else if (part.contains("->") && part.contains("---")) {
            // Mixed mode
            processMixedMode(part, writer, routeId);
        }
    }

    /**
     * Process route part with single mode (driving or walking)
     */
    private static void processSimpleMode(String part, String delimiter, int mode, PrintWriter writer, int routeId) {
        part = part.replace(delimiter, ";");
        String[] arcParts = part.split("[;]");
        for (int arc = 0; arc < arcParts.length - 1; arc++) {
            if (!arcParts[arc].equals(arcParts[arc + 1])) {
                writer.println(arcParts[arc] + ";" + arcParts[arc + 1] + ";" + mode + ";" + routeId);
            }
        }
    }

    /**
     * Process route part with mixed modes (driving and walking)
     */
    private static void processMixedMode(String part, PrintWriter writer, int routeId) {
        // Implementation of mixed mode processing (keeping original complex logic)
        part = part.replace("---", ";").replace("->", ":");

        int posInString = 0;
        String tail = "";
        String head = "";
        int mode = -1;
        int lastPos = -1;

        while (posInString < part.length()) {
            char currentChar = part.charAt(posInString);

            if (mode == -1) {
                if (currentChar == ':') {
                    mode = 1;
                    lastPos = posInString;
                } else if (currentChar == ';') {
                    mode = 2;
                    lastPos = posInString;
                }
            } else {
                if (currentChar == ':' || currentChar == ';') {
                    if (!tail.equals(head)) {
                        writer.println(tail + ";" + head + ";" + mode + ";" + routeId);
                    }
                    posInString = lastPos;
                    mode = -1;
                    tail = "";
                    head = "";
                }
            }

            if (mode == -1 && currentChar != ':' && currentChar != ';') {
                tail += currentChar;
            } else if (currentChar != ':' && currentChar != ';') {
                head += currentChar;
            }

            posInString++;
        }

        if (!tail.equals(head)) {
            writer.println(tail + ";" + head + ";" + mode + ";" + routeId);
        }
    }

}
