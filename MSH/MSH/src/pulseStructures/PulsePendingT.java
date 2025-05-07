package pulseStructures;

public class PulsePendingT {

	/**
	 * Node id on which the pulse is actually on.
	 */
	private String NodeID;

	/**
	 * Pending pulse weights
	 */
	
	private double[] pendingWeights;
	
	/**
	 * Predecessor
	 */
	
	private String predId;
	
	/**
	 * Auxiliary id (For sorting the pending queue)
	 */
	private int auxID;
	
	
	/**
	 * This method creates a pending pulse.
	 * @param id
	 * @param auxId
	 * @param partialWeights 0: cost 1: time 
	 * @param path
	 */
	public PulsePendingT(String id, int auxId,double[] partialWeights, String pred) {
		NodeID = id;
		auxID = auxId;
		pendingWeights = new double[3];
		pendingWeights[0] = partialWeights[0];
		pendingWeights[1] = partialWeights[1];
		pendingWeights[2] = partialWeights[2];
		predId = pred;
	
	}


	/**
	 * @return the nodeID
	 */
	public String getNodeID() {
		return NodeID;
	}


	/**
	 * @param nodeID the nodeID to set
	 */
	public void setNodeID(String nodeID) {
		NodeID = nodeID;
	}


	/**
	 * @return the pendingWeights
	 */
	public double[] getPendingWeights() {
		return pendingWeights;
	}


	/**
	 * @param pendingWeights the pendingWeights to set
	 */
	public void setPendingWeights(double[] pendingWeights) {
		this.pendingWeights = pendingWeights;
	}

	public String getPredId() {
		return predId;
	}


	public void setPredId(String predId) {
		this.predId = predId;
	}


	/**
	 * @return the auxID
	 */
	public int getAuxID() {
		return auxID;
	}


	/**
	 * @param auxID the auxID to set
	 */
	public void setAuxID(int auxID) {
		this.auxID = auxID;
	}



	
	/**
	 * This method returns the walking distance
	 * @return
	 */
	public double getDist() {
		return pendingWeights[2];
	}
	
	/**
	 * This method returns the time 
	 * @return
	 */
	public double getTime() {
		return pendingWeights[1];
	}
	
	/**
	 * This method returns the cost 
	 * @return
	 */
	public double getCost() {
		return pendingWeights[0];
	}
}
