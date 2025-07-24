package validation;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import core.Route;
import core.RouteAttribute;
import core.DistanceMatrix;
import dataStructures.DataHandler;
import distanceMatrices.DepotToCustomersDistanceMatrix;
import distanceMatrices.DepotToCustomersDrivingTimesMatrix;
import distanceMatrices.DepotToCustomersWalkingTimesMatrix;
import globalParameters.GlobalParameters;
import globalParameters.GlobalParametersReader;

/**
 * Classe pour valider si les routes satisfont les contraintes du problème
 * en utilisant les mêmes contraintes que le PulseHandler et en analysant
 * les chaînes de routes comme dans SolutionPrinter
 */
public class RouteConstraintValidator {

    private DataHandler data;
    private DistanceMatrix distances;
    private DistanceMatrix drivingTimes;
    private DistanceMatrix walkingTimes;
    private static Properties parameters;

    /**
     * Constructeur utilisant les mêmes matrices que le solveur
     * 
     * @param instance_identifier identifiant de l'instance
     */
    public RouteConstraintValidator(String instance_identifier, String configpath) throws IOException {
        // Charge les données de la même manière que le solveur
        this.data = new DataHandler(GlobalParameters.INSTANCE_FOLDER + instance_identifier);

        parameters = new Properties();
        FileInputStream fis = null;

        fis = new FileInputStream(configpath);
        parameters.loadFromXML(fis);

        // Initialise les matrices de distances comme dans le solveur
        this.distances = new DepotToCustomersDistanceMatrix(data);
        this.drivingTimes = new DepotToCustomersDrivingTimesMatrix(data);
        this.walkingTimes = new DepotToCustomersWalkingTimesMatrix(data);
    }

    /**
     * Valide une route complète en utilisant les contraintes du PulseHandler
     * 
     * @param route la route à valider
     * @return ValidationResult contenant les résultats de validation
     */
    public ValidationResult validateRoute(Route route) {
        ValidationResult result = new ValidationResult();
        ArrayList<Integer> routeNodes = (ArrayList<Integer>) route.getRoute();

        if (routeNodes.size() < 2) {
            result.isValid = false;
            result.violations.add("Route trop courte (moins de 2 nœuds)");
            return result;
        }

        // Valider en utilisant les contraintes du PulseHandler et l'analyse de chaîne
        result = validateWithChainAnalysis(route, result);

        // Vérifier la cohérence avec les attributs stockés
        result = validateRouteAttributes(route, result);

        result.isValid = result.violations.isEmpty();
        return result;
    }

    /**
     * Valide les contraintes en analysant la chaîne de la route
     * (même logique que SolutionPrinter.generateArcsFromRoute)
     */
    private ValidationResult validateWithChainAnalysis(Route route, ValidationResult result) {
        String chain = (String) route.getAttribute(RouteAttribute.CHAIN);
        if (chain == null) {
            result.violations.add("Route sans chaîne (CHAIN attribute manquant)");
            return result;
        }

        double totalWalkingDistance = 0.0;
        double totalDrivingTime = 0.0;
        double totalWalkingTime = 0.0;
        double totalCost = 0.0;
        double totalServiceTime = 0.0;
        double totalParkingTime = 0.0;

        Set<Integer> visitedNodes = new HashSet<>();

        // Préparer la chaîne comme dans SolutionPrinter
        chain = chain.replace(" 0 ", " " + (0) + " ");
        chain = chain.replace("CD", "" + (0));
        chain = chain.replace(" ", "");
        String[] parts = chain.split("[||]");

        for (int pos = 0; pos < parts.length; pos++) {
            if (parts[pos].length() > 0) {

                // Segment conduit seulement (->)
                if (parts[pos].contains("->") && !parts[pos].contains("---")) {
                    SegmentValidation segmentResult = validateDrivingSegment(parts[pos], visitedNodes);
                    totalDrivingTime += segmentResult.drivingTime;
                    totalServiceTime += segmentResult.serviceTime;
                    totalParkingTime += segmentResult.parkingTime;
                    totalCost += segmentResult.cost;
                    result.violations.addAll(segmentResult.violations);
                }

                // Segment marché seulement (---)
                else if (!parts[pos].contains("->") && parts[pos].contains("---")) {
                    SegmentValidation segmentResult = validateWalkingSegment(parts[pos], visitedNodes);
                    totalWalkingDistance += segmentResult.walkingDistance;
                    totalWalkingTime += segmentResult.walkingTime;
                    totalServiceTime += segmentResult.serviceTime;
                    totalCost += segmentResult.cost;
                    result.violations.addAll(segmentResult.violations);
                }

                // Segment mixte (-> et ---)
                else if (parts[pos].contains("->") && parts[pos].contains("---")) {
                    SegmentValidation segmentResult = validateMixedSegment(parts[pos], visitedNodes);
                    totalWalkingDistance += segmentResult.walkingDistance;
                    totalWalkingTime += segmentResult.walkingTime;
                    totalDrivingTime += segmentResult.drivingTime;
                    totalServiceTime += segmentResult.serviceTime;
                    totalParkingTime += segmentResult.parkingTime;
                    totalCost += segmentResult.cost;
                    result.violations.addAll(segmentResult.violations);
                }
            }
        }

        // Ajouter le coût fixe du véhicule
        ArrayList<Integer> routeNodes = (ArrayList<Integer>) route.getRoute();
        boolean hasCustomerVisits = routeNodes.stream()
                .anyMatch(node -> node > 0 && node <= data.getNbCustomers());

        if (hasCustomerVisits) {
            totalCost += Double.valueOf(parameters.get("FIXED_COST").toString());
        }

        // Calculer la durée totale (marche + conduite + service + stationnement)
        double totalDuration = totalWalkingTime + totalDrivingTime + totalServiceTime + totalParkingTime;

        // Vérifier les contraintes globales
        if (totalWalkingDistance > Double.valueOf(parameters.get("ROUTE_WALKING_DISTANCE_LIMIT").toString())) {
            result.violations.add(String.format(
                    "Distance de marche totale (%.2f) dépasse ROUTE_WALKING_DISTANCE_LIMIT (%.2f)",
                    totalWalkingDistance, Double.valueOf(parameters.get("ROUTE_WALKING_DISTANCE_LIMIT").toString())));
        }

        if (totalDuration > Double.valueOf(parameters.get("ROUTE_DURATION_LIMIT").toString())) {
            result.violations.add(String.format(
                    "Durée totale (%.2f min) dépasse ROUTE_DURATION_LIMIT (%.2f min)",
                    totalDuration, Double.valueOf(parameters.get("ROUTE_DURATION_LIMIT").toString())));
        }

        if (totalDrivingTime > Double.valueOf(parameters.get("SUBTOUR_TIME_LIMIT").toString())) {
            result.violations.add(String.format(
                    "Temps de conduite (%.2f min) dépasse SUBTOUR_TIME_LIMIT (%.2f min)",
                    totalDrivingTime, Double.valueOf(parameters.get("SUBTOUR_TIME_LIMIT").toString())));
        }

        // Stocker les valeurs calculées
        result.totalWalkingDistance = totalWalkingDistance;
        result.totalDuration = totalDuration; // FIXED: Include all time components
        result.totalDrivingTime = totalDrivingTime;
        result.totalWalkingTime = totalWalkingTime;
        result.totalServiceTime = totalServiceTime;
        result.totalParkingTime = totalParkingTime;
        result.totalCost = totalCost;

        return result;
    }

    /**
     * Valide un segment conduit seulement
     */
    private SegmentValidation validateDrivingSegment(String segment, Set<Integer> visitedNodes) {
        SegmentValidation result = new SegmentValidation();

        segment = segment.replace("->", ";");
        String[] arcParts = segment.split("[;]");

        for (int arc = 0; arc < arcParts.length - 1; arc++) {
            if (!arcParts[arc].equals(arcParts[arc + 1])) {
                try {
                    int from = Integer.parseInt(arcParts[arc]);
                    int to = Integer.parseInt(arcParts[arc + 1]);

                    // Add driving time
                    double drivingTime = drivingTimes.getDistance(from, to);
                    result.drivingTime += drivingTime;

                    // Add service time only once per node
                    if (!visitedNodes.contains(from) && from > 0) {
                        result.serviceTime += data.getService_times().get(from);
                        visitedNodes.add(from);
                    }
                    if (!visitedNodes.contains(to) && to > 0) {
                        result.serviceTime += data.getService_times().get(to);
                        visitedNodes.add(to);
                    }

                    // Add parking time only once per segment (at the end)
                    if (arc == arcParts.length - 2) {
                        result.parkingTime += Double.valueOf(parameters.get("PARKING_TIME_MIN").toString());
                    }

                    // Add variable cost based on distance
                    result.cost += distances.getDistance(from, to)
                            * Double.valueOf(parameters.get("VARIABLE_COST").toString());

                } catch (NumberFormatException e) {
                    result.violations
                            .add("Erreur parsing segment conduit: " + arcParts[arc] + " -> " + arcParts[arc + 1]);
                }
            }
        }

        return result;
    }

    /**
     * Valide un segment marché seulement
     */
    private SegmentValidation validateWalkingSegment(String segment, Set<Integer> visitedNodes) {
        SegmentValidation result = new SegmentValidation();

        segment = segment.replace("---", ";");
        String[] arcParts = segment.split("[;]");

        for (int arc = 0; arc < arcParts.length - 1; arc++) {
            try {
                int from = Integer.parseInt(arcParts[arc]);
                int to = Integer.parseInt(arcParts[arc + 1]);

                double walkingDistance = distances.getDistance(from, to);
                double walkingTime = walkingTimes.getDistance(from, to);

                double x_from = data.getX_coors().get(from);
                double y_from = data.getY_coors().get(from);

                // Vérifier MAX_WD_B2P pour chaque arc marché
                if (walkingDistance > Double.valueOf(parameters.get("MAX_WD_B2P").toString())) {
                    result.violations.add(String.format(
                            "Distance de marche entre %d et %d (%.2f) dépasse MAX_WD_B2P (%.2f)",
                            from, to, walkingDistance, Double.valueOf(parameters.get("MAX_WD_B2P").toString())));
                }

                if (GlobalParameters.UPPER_RIGHT_CONSTRAINT && x_from > 5 && y_from > 5) {
                    result.violations.add(String.format(
                            "Arc marché de %d à %d interdit par contrainte supérieure droite (x=%.2f, y=%.2f)",
                            from, to, x_from, y_from));
                }

                result.walkingDistance += walkingDistance;
                result.walkingTime += walkingTime;

                // Add service time only once per node
                if (!visitedNodes.contains(from) && from > 0) {
                    result.serviceTime += data.getService_times().get(from);
                    visitedNodes.add(from);
                }
                if (!visitedNodes.contains(to) && to > 0) {
                    result.serviceTime += data.getService_times().get(to);
                    visitedNodes.add(to);
                }

            } catch (NumberFormatException e) {
                result.violations.add("Erreur parsing segment marché: " + arcParts[arc] + " --- " + arcParts[arc + 1]);
            }
        }

        return result;
    }

    /**
     * Valide un arc spécifique selon son mode (1=conduite, 2=marche)
     */
    private SegmentValidation validateArcWithMode(String tailStr, String headStr, int mode, Set<Integer> visitedNodes) {
        SegmentValidation result = new SegmentValidation();

        try {
            int from = Integer.parseInt(tailStr);
            int to = Integer.parseInt(headStr);

            if (mode == 1) {
                // Arc conduit
                double drivingTime = drivingTimes.getDistance(from, to);
                result.drivingTime += drivingTime;
                result.cost += distances.getDistance(from, to)
                        * Double.valueOf(parameters.get("VARIABLE_COST").toString());

            } else if (mode == 2) {
                // Arc marché
                double walkingDistance = distances.getDistance(from, to);
                double walkingTime = walkingTimes.getDistance(from, to);

                double x_from = data.getX_coors().get(from);
                double y_from = data.getY_coors().get(from);

                // Vérifier MAX_WD_B2P
                if (walkingDistance > Double.valueOf(parameters.get("MAX_WD_B2P").toString())) {
                    result.violations.add(String.format(
                            "Distance de marche entre %d et %d (%.2f) dépasse MAX_WD_B2P (%.2f)",
                            from, to, walkingDistance, Double.valueOf(parameters.get("MAX_WD_B2P").toString())));
                }

                result.walkingDistance += walkingDistance;
                result.walkingTime += walkingTime;

                if (GlobalParameters.UPPER_RIGHT_CONSTRAINT && x_from > 5 && y_from > 5) {
                    result.violations.add(String.format(
                            "Arc marché de %d à %d interdit par contrainte supérieure droite (x=%.2f, y=%.2f)",
                            from, to, x_from, y_from));
                    System.out.println("Arc interdit par contrainte supérieure droite: " + from + " -> " + to);
                }
            }

            // Add service time only once per node
            if (!visitedNodes.contains(from) && from > 0) {
                result.serviceTime += data.getService_times().get(from);
                visitedNodes.add(from);
            }
            if (!visitedNodes.contains(to) && to > 0) {
                result.serviceTime += data.getService_times().get(to);
                visitedNodes.add(to);
            }

        } catch (NumberFormatException e) {
            result.violations.add("Erreur parsing arc: " + tailStr + " -> " + headStr);
        }

        return result;
    }

    /**
     * Valide un segment mixte (conduite + marche)
     * Logique exacte de SolutionPrinter.generateArcsFromRoute
     */
    private SegmentValidation validateMixedSegment(String segment, Set<Integer> visitedNodes) {
        SegmentValidation result = new SegmentValidation();

        segment = segment.replace("---", ";");
        segment = segment.replace("->", ":");

        int posInString = 0;
        String tail = "";
        String head = "";
        int mode = -1; // 1 = conduite, 2 = marche
        int lastPos = -1;
        boolean hasDrivingArcs = false;

        while (posInString < segment.length()) {
            if (mode == -1) {
                if (segment.charAt(posInString) == ':') {
                    mode = 1; // conduite
                    lastPos = posInString;
                    hasDrivingArcs = true;
                }
                if (segment.charAt(posInString) == ';') {
                    mode = 2; // marche
                    lastPos = posInString;
                }
            } else {
                if (segment.charAt(posInString) == ':' || segment.charAt(posInString) == ';') {
                    if (!tail.equals(head)) {
                        // Traiter l'arc tail -> head avec le mode approprié
                        SegmentValidation arcResult = validateArcWithMode(tail, head, mode, visitedNodes);
                        result.addFrom(arcResult);
                    }
                    posInString = lastPos;
                    mode = -1;
                    tail = "";
                    head = "";
                }
            }

            if (mode == -1 && segment.charAt(posInString) != ':' && segment.charAt(posInString) != ';') {
                tail += segment.charAt(posInString);
            } else if (segment.charAt(posInString) != ':' && segment.charAt(posInString) != ';') {
                head += segment.charAt(posInString);
            }

            posInString++;
        }

        // Traiter le dernier arc
        if (!tail.equals(head)) {
            SegmentValidation arcResult = validateArcWithMode(tail, head, mode, visitedNodes);
            result.addFrom(arcResult);
        }

        // Add parking time once per mixed segment if it contains driving arcs
        if (hasDrivingArcs) {
            result.parkingTime += Double.valueOf(parameters.get("PARKING_TIME_MIN").toString());
        }

        return result;
    }

    /**
     * Valide la cohérence avec les attributs stockés dans la route
     */
    private ValidationResult validateRouteAttributes(Route route, ValidationResult result) {
        // Vérifier le coût
        Double routeCost = (Double) route.getAttribute(RouteAttribute.COST);
        if (routeCost != null) {
            if (Math.abs(routeCost - result.totalCost) > Math.pow(10,
                    -Integer.parseInt(parameters.get("PRECISION").toString()))) {
                result.violations.add(String.format(
                        "Coût stocké (%.2f) incohérent avec coût calculé (%.2f)",
                        routeCost, result.totalCost));
            }
        }

        // Vérifier la durée
        Double routeDuration = (Double) route.getAttribute(RouteAttribute.DURATION);
        if (routeDuration != null) {
            if (Math.abs(routeDuration - result.totalDuration) > (Math.pow(10,
                    -Integer.parseInt(parameters.get("PRECISION").toString())))) {
                result.violations.add(String.format(
                        "Durée stockée (%.2f min) incohérente avec durée calculée (%.2f min)",
                        routeDuration, result.totalDuration));
            }
        }

        return result;
    }

    /**
     * Valide une liste de routes
     */
    public ArrayList<ValidationResult> validateRoutes(ArrayList<Route> routes) {
        ArrayList<ValidationResult> results = new ArrayList<>();

        for (Route route : routes) {
            results.add(validateRoute(route));
        }

        return results;
    }

    /**
     * Valide toutes les contraintes globales d'une solution
     */
    public ValidationResult validateSolution(ArrayList<Route> routes) {
        ValidationResult globalResult = new ValidationResult();

        // Vérifier que tous les clients sont visités exactement une fois
        boolean[] visitedCustomers = new boolean[data.getNbCustomers() + 1];

        for (Route route : routes) {
            ArrayList<Integer> routeNodes = (ArrayList<Integer>) route.getRoute();

            for (int node : routeNodes) {
                if (node > 0 && node <= data.getNbCustomers()) {
                    if (visitedCustomers[node]) {
                        globalResult.violations.add("Client " + node + " visité plusieurs fois");
                    } else {
                        visitedCustomers[node] = true;
                    }
                }
            }
        }

        // Vérifier que tous les clients sont visités
        for (int i = 1; i <= data.getNbCustomers(); i++) {
            if (!visitedCustomers[i]) {
                globalResult.violations.add("Client " + i + " non visité");
            }
        }

        // Calculer le coût total et agréger les violations
        double totalSolutionCost = 0.0;
        for (int i = 0; i < routes.size(); i++) {
            ValidationResult routeResult = validateRoute(routes.get(i));
            totalSolutionCost += routeResult.totalCost;

            // Ajouter les violations de routes individuelles
            for (String violation : routeResult.violations) {
                globalResult.violations.add("Route " + i + ": " + violation);
            }
        }

        globalResult.totalCost = totalSolutionCost;
        globalResult.isValid = globalResult.violations.isEmpty();

        return globalResult;
    }

    /**
     * Renvoie un boolean indiquant si la solution respecte les contraintes globales
     */
    public boolean validateGlobalConstraints(ArrayList<Route> routes) {
        ValidationResult result = validateSolution(routes);
        return result.isValid;
    }

    /**
     * Classe interne pour stocker les résultats de validation d'un segment
     */
    private static class SegmentValidation {
        double walkingDistance = 0.0;
        double walkingTime = 0.0;
        double drivingTime = 0.0;
        double serviceTime = 0.0;
        double parkingTime = 0.0;
        double cost = 0.0;
        ArrayList<String> violations = new ArrayList<>();

        void addFrom(SegmentValidation other) {
            this.walkingDistance += other.walkingDistance;
            this.walkingTime += other.walkingTime;
            this.drivingTime += other.drivingTime;
            this.serviceTime += other.serviceTime;
            this.parkingTime += other.parkingTime;
            this.cost += other.cost;
            this.violations.addAll(other.violations);
        }
    }

    /**
     * Classe pour stocker les résultats de validation
     */
    public static class ValidationResult {
        public boolean isValid = true;
        public ArrayList<String> violations = new ArrayList<>();
        public double totalCost = 0.0;
        public double totalWalkingDistance = 0.0;
        public double totalDuration = 0.0;
        public double totalDrivingTime = 0.0;
        public double totalWalkingTime = 0.0;
        public double totalServiceTime = 0.0;
        public double totalParkingTime = 0.0;

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== VALIDATION ROUTE ===\n");
            sb.append("Statut: ").append(isValid ? "VALIDE" : "INVALIDE").append("\n");
            sb.append("\n--- MÉTRIQUES ---\n");
            sb.append("Coût total: ").append(String.format("%.2f", totalCost)).append("\n");
            sb.append("Distance MARCHÉE: ").append(String.format("%.2f", totalWalkingDistance))
                    .append(" / ").append(Double.valueOf(parameters.get("ROUTE_WALKING_DISTANCE_LIMIT").toString()))
                    .append(" km\n");
            sb.append("Durée TOTALE: ").append(String.format("%.2f", totalDuration))
                    .append(" / ").append(Double.valueOf(parameters.get("ROUTE_DURATION_LIMIT").toString()))
                    .append(" min\n");
            sb.append("  - Temps marche: ").append(String.format("%.2f", totalWalkingTime)).append(" min\n");
            sb.append("  - Temps conduite: ").append(String.format("%.2f", totalDrivingTime)).append(" min\n");
            sb.append("  - Temps service: ").append(String.format("%.2f", totalServiceTime)).append(" min\n");
            sb.append("  - Temps stationnement: ").append(String.format("%.2f", totalParkingTime)).append(" min\n");

            if (!violations.isEmpty()) {
                sb.append("\n--- VIOLATIONS (").append(violations.size()).append(") ---\n");
                for (String violation : violations) {
                    sb.append(violation).append("\n");
                }
            } else {
                sb.append("\nAucune violation détectée\n");
            }

            return sb.toString();
        }

        public String toSummary() {
            return String.format(
                    "Validation: %s | Violations: %d | Coût: %.2f | Marche: %.2f/%.2f km | Durée: %.2f/%.2f min",
                    isValid ? "OK" : "KO",
                    violations.size(),
                    totalCost,
                    totalWalkingDistance, Double.valueOf(parameters.get("ROUTE_WALKING_DISTANCE_LIMIT").toString()),
                    totalDuration, Double.valueOf(parameters.get("ROUTE_DURATION_LIMIT").toString()));
        }

        public int getViolationCount() {
            return violations.size();
        }
    }

}