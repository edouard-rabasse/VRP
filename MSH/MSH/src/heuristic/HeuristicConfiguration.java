package heuristic;

import globalParameters.GlobalParameters;

public class HeuristicConfiguration {
    public enum HeuristicConfig {
        HIGH_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_HIGH, GlobalParameters.MSH_RANDOM_FACTOR_HIGH_RN),
        LOW_RANDOMIZATION(GlobalParameters.MSH_RANDOM_FACTOR_LOW, GlobalParameters.MSH_RANDOM_FACTOR_LOW);

        public final int randomFactor;
        public final int nnRandomFactor;

        HeuristicConfig(int randomFactor, int nnRandomFactor) {
            this.randomFactor = randomFactor;
            this.nnRandomFactor = nnRandomFactor;
        }
    }

}
