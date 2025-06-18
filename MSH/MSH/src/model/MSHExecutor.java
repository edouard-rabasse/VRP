package model;

import globalParameters.GlobalParameters;
import msh.AssemblyFunction;
import msh.MSH;

/**
 * Handles the execution timing and workflow for MSH optimization phases.
 * This class encapsulates the sampling and assembly phases with proper timing
 * measurements.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class MSHExecutor {

    private double cpuSampling;
    private double cpuAssembly;

    /**
     * Executes the complete MSH workflow including sampling and assembly phases
     */
    public MSHExecutionResult execute(MSH msh) {
        // Sampling phase
        double samplingTime = executeSamplingPhase(msh);

        // Assembly phase
        double assemblyTime = executeAssemblyPhase(msh);

        return new MSHExecutionResult(samplingTime, assemblyTime);
    }

    /**
     * Executes only the sampling phase
     */
    public double executeSamplingPhase(MSH msh) {
        double startTime = getCurrentTime();

        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println("Start of the sampling step...");
        }

        msh.run_sampling();

        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println("End of the sampling step...");
        }

        double endTime = getCurrentTime();
        this.cpuSampling = (endTime - startTime) / 1000000000;

        return this.cpuSampling;
    }

    /**
     * Executes only the assembly phase
     */
    public double executeAssemblyPhase(MSH msh) {
        double startTime = getCurrentTime();

        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println("Start of the assembly step...");
        }

        msh.run_assembly();

        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println("End of the assembly step...");
        }

        double endTime = getCurrentTime();
        this.cpuAssembly = (endTime - startTime) / 1000000000;

        return this.cpuAssembly;
    }

    /**
     * Runs the MSH configuration, executing both sampling and assembly phases,
     * and returns the MSH result.
     */
    public MSHResult runMSH(MSHConfiguration config) {
        // Initialization time (no heavy setup here, so set to 0)
        double initTime = 0.0;
        MSH msh = config.getMSH();
        AssemblyFunction assembler = config.getAssembler();
        // Sampling and assembly phases
        double samplingTime = executeSamplingPhase(msh);
        double assemblyTime = executeAssemblyPhase(msh);
        return new MSHResult(msh, assembler, initTime, samplingTime, assemblyTime);
    }

    private double getCurrentTime() {
        return (double) System.nanoTime();
    }

    // Getters
    public double getCpuSampling() {
        return cpuSampling;
    }

    public double getCpuAssembly() {
        return cpuAssembly;
    }

    public double getTotalTime() {
        return cpuSampling + cpuAssembly;
    }

    /**
     * Result container for MSH execution
     */
    public static class MSHExecutionResult {
        private final double samplingTime;
        private final double assemblyTime;

        public MSHExecutionResult(double samplingTime, double assemblyTime) {
            this.samplingTime = samplingTime;
            this.assemblyTime = assemblyTime;
        }

        public double getSamplingTime() {
            return samplingTime;
        }

        public double getAssemblyTime() {
            return assemblyTime;
        }

        public double getTotalTime() {
            return samplingTime + assemblyTime;
        }
    }
}
