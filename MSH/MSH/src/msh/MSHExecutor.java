package msh;

import globalParameters.GlobalParameters;

public class MSHExecutor {
    /**
     * Execute only MSH phases (sampling and assembly)
     */
    public static void executeMSHPhases(MSH msh) {
        // Sampling phase
        executeSampling(msh);

        // Assembly phase
        executeAssembly(msh);
    }

    /**
     * Execute the sampling phase of MSH
     * 
     * @param msh
     * @return cpu_msh_sampling time in seconds
     */
    public static double executeSampling(MSH msh) {
        Double iniTime = (double) System.nanoTime();
        printMessage("Start of the sampling step...");
        msh.run_sampling();
        printMessage("End of the sampling step...");
        Double finTime = (double) System.nanoTime();
        double cpu_msh_sampling = (finTime - iniTime) / 1000000000;
        return cpu_msh_sampling;
    }

    /**
     * Execute the assembly phase of MSH
     * 
     * @param msh
     * @return cpu_msh_assembly time in seconds
     */
    public static double executeAssembly(MSH msh) {
        Double iniTime = (double) System.nanoTime();
        printMessage("Start of the assembly step...");
        msh.run_assembly();
        printMessage("End of the assembly step...");
        Double finTime = (double) System.nanoTime();
        double cpu_msh_assembly = (finTime - iniTime) / 1000000000;
        return cpu_msh_assembly;
    }

    /**
     * Print message if console printing is enabled
     */
    public static void printMessage(String message) {
        if (GlobalParameters.PRINT_IN_CONSOLE) {
            System.out.println(message);
        }
    }

}
