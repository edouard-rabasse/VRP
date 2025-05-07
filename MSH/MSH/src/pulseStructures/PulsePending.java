package pulseStructures;

/**
 * This class represents a pulse that has been put on hold.
 * @author nick0
 *
 */
public class PulsePending {

	/**
	 * Node id on which the pulse is actually on.
	 */
	private String NodeID;

	/**
	 * Pending pulse weights
	 */
	
	private double[] pendingWeights;
	
	/**
	 * False if the pulse is already treated.
	 */
	private boolean notTreated;
	
	/**
	 * Predecessor
	 */
	
	private String predId;
	
	/**
	 * Auxiliary id (For sorting the pending queue)
	 */
	private int auxID;
	
	/**
	 * Sort criteria - Best promise
	 */
	private double sortCriteria;
	
	
	/**
	 * This method creates a pending pulse.
	 * @param id
	 * @param auxId
	 * @param partialWeights 0: cost 1: time 
	 * @param path
	 */
	public PulsePending(String id, int auxId,double[] partialWeights, String pred) {
		NodeID = id;
		auxID = auxId;
		pendingWeights = new double[3];
		pendingWeights[0] = partialWeights[0];
		pendingWeights[1] = partialWeights[1];
		pendingWeights[2] = partialWeights[2];
		notTreated = true;
		sortCriteria = 0;
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


	/**
	 * @return the notTreated
	 */
	public boolean getNotTreated() {
		return notTreated;
	}


	/**
	 * @param notTreated the notTreated to set
	 */
	public void setNotTreated(boolean notTreated) {
		this.notTreated = notTreated;
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
	 * @return the sortCriteria
	 */
	public double getSortCriteria() {
		return sortCriteria;
	}


	/**
	 * @param sortCriteria the sortCriteria to set
	 */
	public void setSortCriteria(double sortCriteria) {
		this.sortCriteria = sortCriteria;
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
