package util;

/**
 * Data class to hold cost breakdown information
 * 
 * @author Edouard Rabasse
 */
public class CostBreakdown {
    private final double realCost;
    private final double customCost;
    private final double penalty;

    public CostBreakdown(double realCost, double customCost, double penalty) {
        this.realCost = realCost;
        this.customCost = customCost;
        this.penalty = penalty;
    }

    // Getters
    public double getRealCost() {
        return realCost;
    }

    public double getCustomCost() {
        return customCost;
    }

    public double getPenalty() {
        return penalty;
    }

    public double getPenaltyPercentage() {
        return realCost > 0 ? (penalty / realCost * 100) : 0;
    }

    @Override
    public String toString() {
        return String.format("CostBreakdown{real=%.2f, custom=%.2f, penalty=%.2f (%.1f%%)}",
                realCost, customCost, penalty, getPenaltyPercentage());
    }
}