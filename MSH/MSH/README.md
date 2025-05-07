===========================================================================================================================================================================
 Readme for the MSH Java code for solving the PLRP
 Version: 1.0
===========================================================================================================================================================================

 Author:       Nicolás Cabrera (nicolas.cabrera-malik@hec.ca)
               Logistics and Operations Management department
               HEC Montreal       

===========================================================================================================================================================================

This file contains important information about the Java code for the PLRP.

To visualize all our solutions or new ones you can use this website: https://nicolascabrera.shinyapps.io/PLRP/

===========================================================================================================================================================================

The file SPLRP contains all the source code for executing the Multi-space sampling heuristic (MSH) for solving the park-and-loop routing problem (PLRP). 

The main class is called "Main". In this class the user can select the instance (line 39) and the configuration file (line 43).

The configuration file is an xml file that should be saved in the "config" folder. 
 
We include a sample configuration file (default.xml).


===========================================================================================================================================================================
Default instance
===========================================================================================================================================================================

The default instance is instance 1. The objective function of the best-known solution for this instance is 630.94. The total driving distance is 30.94. The fixed cost is 600.


===========================================================================================================================================================================
References
===========================================================================================================================================================================

-Mendoza, J. E., & Villegas, J. G. (2013). A multi-space sampling heuristic for the vehicle routing problem with stochastic demands. Optimization Letters, 7, 1503-1516.

-Cabrera, N., Cordeau, J. F., & Mendoza, J. E. (2022). The doubly open park-and-loop routing problem. Computers & Operations Research, 143, 105761.

===========================================================================================================================================================================
Usage & License
===========================================================================================================================================================================

If you use this code please send a line to nicolas.cabrera-malik@hec.ca describing us your application.
