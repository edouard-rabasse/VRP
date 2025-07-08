package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

import core.JVRAEnv;
import core.Route;
import core.RouteAttribute;
import dataStructures.DataHandler;

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

    public static int countRoutesInFile(String arcPath) throws IOException {
        int numRoutes = 0;
        try (BufferedReader reader = new BufferedReader(new FileReader(arcPath))) {
            String line;
            int lastRouteId = -1;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(";");
                int currentRoute = Integer.parseInt(parts[3]);
                if (currentRoute != lastRouteId) {
                    numRoutes++;
                    lastRouteId = currentRoute;
                }
            }
        }
        return numRoutes;
    }

    public static Route convertRouteToGlobalRoute(Route route, DataHandler localData, DataHandler globalData) {
        Route r_copy = JVRAEnv.getRouteFactory().buildRoute();
        r_copy.add(0);
        for (int i = 1; i < route.size() - 1; i++) {

            int customer = route.get(i);
            int mapped = localData.getMapping().get(customer);
            r_copy.add(mapped);
        }
        r_copy.add(0);
        r_copy.setAttribute(RouteAttribute.COST, route.getAttribute(RouteAttribute.COST));
        r_copy.setAttribute(RouteAttribute.DURATION, route.getAttribute(RouteAttribute.DURATION));

        // Convert chain string
        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
        Pattern numberPattern = Pattern.compile("\\b\\d+\\b");
        Matcher matcher = numberPattern.matcher(chain);
        StringBuffer result = new StringBuffer();
        while (matcher.find()) {
            int original = Integer.parseInt(matcher.group());
            int mapped = localData.getMapping().getOrDefault(original, original);
            matcher.appendReplacement(result, String.valueOf(mapped));
        }
        matcher.appendTail(result);
        r_copy.setAttribute(RouteAttribute.CHAIN, result.toString());

        return r_copy;
    }

}
