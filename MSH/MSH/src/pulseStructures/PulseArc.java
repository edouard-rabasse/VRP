package pulseStructures;

/**
 * This class represents an arc on the "Pulse" graph.
 * @author nick0
 *
 */
public class PulseArc {
	/**
	 * The arc tail
	 */
	private PulseNode tail;
	
	/**
	 * The arc head
	 */
	private PulseNode head;
	
	/**
	 * The arc time
	 */
	private double time;
	
	/**
	 * The arc cost
	 */
	private double cost;
	
	/**
	 * The arc walking time
	 */
	private double walk_time;
	
	
	/**
	 * The sort criteria
	 */
	private int sortCriteria;

	/**
	 * This method creates an arc for the pulse procedure
	 * @param ta tail node
	 * @param he head node
	 * @param co arc cost
	 * @param ti arc time
	 */
	public PulseArc(PulseNode ta, PulseNode he, double co, double ti, double di) {
		tail = ta;
		head = he;
		cost = co;
		time = ti;
		walk_time = di;
		he.getIncomingArcs().add(this);
		ta.getOutgoingArcs().add(this);
	}

	/**
	 * @return the tail
	 */
	public PulseNode getTail() {
		return tail;
	}

	/**
	 * @param tail the tail to set
	 */
	public void setTail(PulseNode tail) {
		this.tail = tail;
	}

	/**
	 * @return the head
	 */
	public PulseNode getHead() {
		return head;
	}

	/**
	 * @param head the head to set
	 */
	public void setHead(PulseNode head) {
		this.head = head;
	}

	/**
	 * @return the cost
	 */
	public double getCost() {
		return cost;
	}

	/**
	 * @param cost the cost to set
	 */
	public void setCost(int cost) {
		this.cost = cost;
	}

	/**
	 * @return the time
	 */
	public double getTime() {
		return time;
	}

	/**
	 * @param time the time to set
	 */
	public void setTime(int time) {
		this.time = time;
	}

	/**
	 * @return the sortCriteria
	 */
	public int getSortCriteria() {
		return sortCriteria;
	}

	/**
	 * @param sortCriteria the sortCriteria to set
	 */
	public void setSortCriteria(int sortCriteria) {
		this.sortCriteria = sortCriteria;
	}

	/**
	 * @return the walk_time
	 */
	public double getWalk_time() {
		return walk_time;
	}

	/**
	 * @param walk_time the walk_time to set
	 */
	public void setWalk_time(double walk_time) {
		this.walk_time = walk_time;
	}

	
}
