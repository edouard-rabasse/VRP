package model;

import msh.MSH;
import msh.AssemblyFunction;

/**
 * Encapsulates the results of running the MSH algorithm.
 * This class holds all the timing information and results from an MSH
 * execution.
 * 
 * @author Refactored by Edouard Rabasse
 */
public class MSHResult {

    private final MSH msh;
    private final AssemblyFunction assembler;
    private final double initializationTime;
    private final double samplingTime;
    private final double assemblyTime;

    public MSHResult(MSH msh, AssemblyFunction assembler, double initializationTime,
            double samplingTime, double assemblyTime) {
        this.msh = msh;
        this.assembler = assembler;
        this.initializationTime = initializationTime;
        this.samplingTime = samplingTime;
        this.assemblyTime = assemblyTime;
    }

    public MSH getMsh() {
        return msh;
    }

    public AssemblyFunction getAssembler() {
        return assembler;
    }

    public double getInitializationTime() {
        return initializationTime;
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
