package pulseStructures;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * This class represents a node in the pulse graph. Here you will find the function pulse and pulse with queues.
 * @author nick0
 *
 */
public class PulseNode {

	/**
	 * The node id
	 */
	private String id;
	 
	/**
	 * The list of incoming arcs
	 */
	private ArrayList<PulseArc> incomingArcs;
	
	/**
	 * The list of outgoing arcs
	 */
	private ArrayList<PulseArc> outgoingArcs;
	
	//Node SP labels
	 
	/**
	 * Minimum cost path to the final node
	 */
	private double minCost[];
	
	/**
	 * Minimum time path to the final node
	 */
	private double minTime[];
	
	/**
	 * Minimum walking distance path to the final node
	 */
	private double minDist[];
	
	/**
	 * Boolean that tells if the node is visited for first time
	 */
	boolean firstTime = true;
	
	/**
	 * Labels for dominance prunning
	 */
	
	private ArrayList<Label> labels;
	
	/**
	 * ArrayList of pending pulses
	 */
	
	public ArrayList<PulsePending> pend;
	
	/**
	 * ArrayList of all pending pulses
	 */
	
	public ArrayList<PulsePendingT> pendT;
	
	
	/**
	 * Auxiliary id for the pending queue sorting
	 */
	
	public int auxID;
	
	
	/**
	 * Creates a new node
	 * @param i
	 */
	public PulseNode(String i,int nI) {
		id = i;
		pend = new ArrayList<PulsePending>();
		pendT = new ArrayList<PulsePendingT>();
		incomingArcs = new ArrayList<PulseArc>();
		outgoingArcs = new ArrayList<PulseArc>();
		auxID = nI;
		setLabels();
	}
	
	/**
	 * This is the pulse algorithm
	 * @param pulseWeights: 0 cost 1 time 2 dist
	 * @param step
	 * @param path
	 */
	public void pulse(double[] pulseWeights,int step, String pred,PulseHandler pH) {
		
		//If node is visited for the first time: Sort the arcs.
		
			if (this.firstTime) {
				this.firstTime = false;
				this.Sort(incomingArcs);
			}
			
		//Check if the pulse can be pruned by dominance
			this.pendT.add(new PulsePendingT(this.id,this.auxID,pulseWeights,pred));
			
			if(notDominated(pulseWeights,this,pH) && !pH.isStop()) {
				
				//Creates a pending pulse:
				
					PulsePending p = new PulsePending(this.id,this.auxID,pulseWeights,pred);
					
					p.setSortCriteria(pulseWeights[0] + this.getMinCost()[0]);
					p.setNotTreated(false); //Note that this pulse is not added to the global pending queue, so is "treated"
					this.pend.add(p);
					
				//Check if we can safely complete the path. MT or MC
					
					if(completePathCheck(pulseWeights,pH) && !pH.isStop()){
						
							
						//Check the incoming arcs (Remember we are going backwards!)
							
							for(Iterator<PulseArc> iter = incomingArcs.iterator();iter.hasNext();) {
								
								//Relevant info for the arc and the tail
								
									PulseArc a = iter.next();
									double[] newWeights = new double[3];
									newWeights[0] = (pulseWeights[0] + a.getCost()); //Cost after adding the node	
									newWeights[1] = (pulseWeights[1] + a.getTime()); //Time after adding the node
									newWeights[2] = (pulseWeights[2] + a.getWalk_time()); //Time after adding the node

									PulseNode tail = a.getTail(); //Tail of the arc
								
								//Check for infeasibility and bound pruning
									
								if( (newWeights[2] + tail.getMinDist()[2] <= pH.getDistLimit()) && (newWeights[1] + tail.getMinTime()[1] <= pH.getTimeLimit()) && (newWeights[0] + tail.getMinCost()[0] < pH.getPrimalBound()) && !pH.isStop()) {
											
									//Update the depth
									
										step++;
									
									//Check if we should stop the pulse and add it to the queue
										
										if(step >= pH.getDepth()) {
											
											//Check if the pulse is dominated before adding it to the queue
												tail.pendT.add(new PulsePendingT(tail.getId(),tail.getAuxID(),newWeights,id));
												if(notDominated(newWeights,tail,pH) && !pH.isStop()) {
													
													//Adds the pulse to the queue
													
														p = new PulsePending(tail.getId(),tail.getAuxID(),newWeights,id);
														p.setSortCriteria(newWeights[0] + tail.getMinCost()[0]);
														pH.addPendingPulse_DOrder(p, pH.getPendingQueue());
														tail.pend.add(p);
		
												}
										}
										else {
											tail.pulse(newWeights, step,id,pH); //Continues the pulse on the tail node
										}
									
									//Update the depth
										
										step--;
									
								}	
							}
						

					}
			}
	}
	
	/**
	 * This method checks if the current pulse is dominated. Also, it eliminates dominated pulses.
	 * @param pulseWeights
	 * @param nodeID
	 * @return
	 */
	public boolean notDominated(double[] pulseWeights,PulseNode node,PulseHandler pH) {
		boolean notdominated = true;
		PulsePending n;
		int rIndex;
		
		if(!pH.isStop()) {
			for(int i = node.pend.size()-1; i >=0 ;i--){
				n = node.pend.get(i);
				if((n.getCost()<=pulseWeights[0] && n.getTime() <= pulseWeights[1] && n.getDist() <= pulseWeights[2])){
					notdominated = false;
					}
				
				else if (n.getCost()>pulseWeights[0] && n.getTime() >= pulseWeights[1] && n.getDist() >= pulseWeights[2]  && pH.getPendingQueue().size() > 0){
					
					if(n.getNotTreated()){
						rIndex = pH.binarySearch(n,pH.getPendingQueue());
						pH.getPendingQueue().remove(rIndex);
					}
			
					node.pend.remove(i); 
					n = null;
				}
				
			}
		}
		
	
		return notdominated;
	}
	
	/**
	 * This is the pulse algorithm
	 * @param pulseWeights: 0 cost 1 time 2 dist
	 * @param step
	 * @param path
	 */
	public void pulseWithQueues(double[] pulseWeights,int step, String pred,PulseHandler pH) {
		
		//If node is visited for the first time: Sort the arcs.
		
			if (this.firstTime) {
				this.firstTime = false;
				this.Sort(incomingArcs);
			}
			
		//Check if we can safely complete the path. MT or MC
			
			if(completePathCheck(pulseWeights,pH) && !pH.isStop()){	
				
					
				//Check the incoming arcs (Remember we are going backwards!)
					
					for(Iterator<PulseArc> iter = incomingArcs.iterator();iter.hasNext();) {
						
						//Relevant info for the arc and the tail
						
							PulseArc a = iter.next();
							double[] newWeights = new double[3];
							newWeights[0] = (pulseWeights[0] + a.getCost()); //Cost after adding the node	
							newWeights[1] = (pulseWeights[1] + a.getTime()); //Time after adding the node
							newWeights[2] = (pulseWeights[2] + a.getWalk_time()); //Time after adding the node
							
							PulseNode tail = a.getTail();
							
						//Check for infeasibility and bound pruning
							
							if((newWeights[2] + tail.getMinDist()[2] <= pH.getDistLimit()) &&  (newWeights[1] + tail.getMinTime()[1] <= pH.getTimeLimit()) && (newWeights[0] + tail.getMinCost()[0] < pH.getPrimalBound()) && !pH.isStop()) {
										
								//Update the depth
								
									step++;
								
								//Check if we should stop the pulse and add it to the queue
									
									if(step >= pH.getDepth()) {
										
										//Check if the pulse is dominated before adding it to the queue
											tail.pendT.add(new PulsePendingT(tail.getId(),tail.getAuxID(),newWeights,id));
											if(notDominated(newWeights,tail,pH) && !pH.isStop()) {
												
												//Adds the pulse to the queue
												
													PulsePending p = new PulsePending(tail.getId(),tail.getAuxID(),newWeights,id);
													p.setSortCriteria(newWeights[0] + tail.getMinCost()[0]);
												    pH.addPendingPulse_DOrder(p, pH.getPendingQueue());
													tail.pend.add(p);
													
											}
									}
									else {
										tail.pulse(newWeights, step,id,pH);
									}
									
								//Update the depth
								
									step--;
							}
					}
					
				
			}
			
	}
	
	
	/**
	 * Path completion
	 * @param pulseWeights: [0] cost [1] time [2] dist
	 * @return true if the search must go on, false if the pulse is stopped
	 */
	public boolean completePathCheck(double[] pulseWeights,PulseHandler pH){
		if(!pH.isStop()) {
			if(pulseWeights[2]+ this.getMinCost()[2] <= pH.getDistLimit() && pulseWeights[1]+ this.getMinCost()[1] <= pH.getTimeLimit()  && pulseWeights[0] + this.getMinCost()[0] < pH.getPrimalBound()){
				pH.setPrimalBound(pulseWeights[0] + this.getMinCost()[0]);
				pH.setTimeStar(pulseWeights[1]+ this.getMinCost()[1]);
				pH.setDistStar(pulseWeights[2]+ this.getMinCost()[2]);
				pH.setBest(2);
				pH.setEndNode(this.id);
				pH.setFinalCost(pulseWeights[0]);
				pH.setFinalTime(pulseWeights[1]);
				pH.setFinalDist(pulseWeights[2]);
	
				pH.setIni(false);
				return false;
			}
			else{
				if(pulseWeights[2]+ this.getMinTime()[2] <= pH.getDistLimit() &&pulseWeights[1]+ this.getMinTime()[1] <= pH.getTimeLimit() && pulseWeights[0] + this.getMinTime()[0] < pH.getPrimalBound()){
	
					pH.setPrimalBound(pulseWeights[0] + this.getMinTime()[0]);
					pH.setTimeStar(pulseWeights[1]+ this.getMinTime()[1]);
					pH.setDistStar(pulseWeights[2]+ this.getMinTime()[2]);
					pH.setBest(1);
					pH.setEndNode(this.id);
					pH.setFinalCost(pulseWeights[0]);
					pH.setFinalTime(pulseWeights[1]);
					pH.setFinalDist(pulseWeights[2]);
					
					pH.setIni(false);

			
				}
				if(pulseWeights[2]+ this.getMinDist()[2] <= pH.getDistLimit() && pulseWeights[1]+ this.getMinDist()[1] <= pH.getTimeLimit() && pulseWeights[0] + this.getMinDist()[0] < pH.getPrimalBound()){
					pH.setPrimalBound(pulseWeights[0] + this.getMinDist()[0]);
					pH.setTimeStar(pulseWeights[1]+ this.getMinDist()[1]);
					pH.setDistStar(pulseWeights[2]+ this.getMinDist()[2]);
					pH.setBest(3);
					pH.setEndNode(this.id);
					pH.setFinalCost(pulseWeights[0]);
					pH.setFinalTime(pulseWeights[1]);
					pH.setFinalDist(pulseWeights[2]);
					
					pH.setIni(false);
				
					//PulseHandler.stop = true;
				}
			}
		}	
		return true;
	}
	
	
	
	
	

	
		/**
		 * This method adds an incoming arc
		 * @param a
		 */
		public void addIncArc(PulseArc a) {
			incomingArcs.add(a);
		}
	
		/**
		 * @return the id
		 */
		public String getId() {
			return id;
		}
	
		/**
		 * @param id the id to set
		 */
		public void setId(String id) {
			this.id = id;
		}
	
	
		/**
		 * @return the incomingArcs
		 */
		public ArrayList<PulseArc> getIncomingArcs() {
			return incomingArcs;
		}
	
		/**
		 * @param incomingArcs the incomingArcs to set
		 */
		public void setIncomingArcs(ArrayList<PulseArc> incomingArcs) {
			this.incomingArcs = incomingArcs;
		}
		
		public void setLabels() {
			minCost = new double[3];
			minTime = new double[3];
			minDist = new double[3];
			for(int i=0;i<=2;i++) {
				minCost[i] = Double.POSITIVE_INFINITY;
				minTime[i] = Double.POSITIVE_INFINITY;
				minDist[i] = Double.POSITIVE_INFINITY;
			}
	
		}
	
		public void setLabels2() {
			for(int i=0;i<=2;i++) {
				minCost[i] = 0;
				minTime[i] = 0;
				minDist[i] = 0;
			}
		}
		
		/**
		 * Sets the labels for the initial node coming from the CD by car
		 */
		public void setLabelsCD(double  cost, double time) {
			minCost[0] = cost;
			minTime[0] = cost;
			minDist[0] = cost;
			
			minCost[1] = time;
			minTime[1] = time;
			minDist[1] = time;
			
			minCost[2] = 0;
			minTime[2] = 0;
			minDist[2] = 0;
			
		}
		/**
		 * @return the minCost
		 */
		public double[] getMinCost() {
			return minCost;
		}

		/**
		 * @param minCost the minCost to set
		 */
		public void setMinCost(double[] minCost) {
			this.minCost = minCost;
		}

		/**
		 * @return the minTime
		 */
		public double[] getMinTime() {
			return minTime;
		}


		/**
		 * @param minTime min time to set
		 */
		public void setMinTime(double[] minTime) {
			this.minTime = minTime;
		}
		
		/**
		 * @return the minDist
		 */
		public double[] getMinDist() {
			return minDist;
		}


		/**
		 * @param minDist min dist to set
		 */
		public void setMinDist(double[] minDist) {
			this.minDist = minDist;
		}
		
		/**
		 * This method sorts the incoming arcs
		 * @param set
		 */
		private void Sort(ArrayList<PulseArc> set) 
		{
			QS(incomingArcs, 0, incomingArcs.size()-1);
		}
		
		/**
		 * 
		 * @param e
		 * @param b
		 * @param t
		 */
		public void QS(ArrayList<PulseArc> e, int b, int t)
		{
			 int pivote;
		     if(b < t){
		        pivote=colocar(e,b,t);
		        QS(e,b,pivote-1);
		        QS(e,pivote+1,t);
		     }  
		}
		
		
		/**
		 * 
		 * @param e
		 * @param b
		 * @param t
		 * @return
		 */
		public int colocar(ArrayList<PulseArc> e, int b, int t)
		{
		    int i;
		    int pivote;
		    double valor_pivote;
		    PulseArc temp;
		    //System.out.println("Encontre  al colocar");
		    pivote = b;
		    valor_pivote = e.get(pivote).getTail().getMinCost()[0];
		    for (i=b+1; i<=t; i++){
		        if (e.get(i).getTail().getMinCost()[0] < valor_pivote){
		                pivote++;    
		                temp= e.get(i);
		                e.set(i, e.get(pivote));
		                e.set(pivote, temp);
		        }
		    }
		    temp=e.get(b);
		    e.set(b, e.get(pivote));
	        e.set(pivote, temp);
		    return pivote;
		    
		}

		/**
		 * @return the firstTime
		 */
		public boolean isFirstTime() {
			return firstTime;
		}

		/**
		 * @param firstTime the firstTime to set
		 */
		public void setFirstTime(boolean firstTime) {
			this.firstTime = firstTime;
		}

		/**
		 * @return the labels
		 */
		public ArrayList<Label> getLabels() {
			return labels;
		}

		/**
		 * @param labels the labels to set
		 */
		public void setLabels(ArrayList<Label> labels) {
			this.labels = labels;
		}

		/**
		 * @return the pend
		 */
		public ArrayList<PulsePending> getPend() {
			return pend;
		}

		/**
		 * @param pend the pend to set
		 */
		public void setPend(ArrayList<PulsePending> pend) {
			this.pend = pend;
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
		 * @return the outgoingArcs
		 */
		public ArrayList<PulseArc> getOutgoingArcs() {
			return outgoingArcs;
		}

		/**
		 * @param outgoingArcs the outgoingArcs to set
		 */
		public void setOutgoingArcs(ArrayList<PulseArc> outgoingArcs) {
			this.outgoingArcs = outgoingArcs;
		}

		/**
		 * @return the pendT
		 */
		public ArrayList<PulsePendingT> getPendT() {
			return pendT;
		}

		/**
		 * @param pendT the pendT to set
		 */
		public void setPendT(ArrayList<PulsePendingT> pendT) {
			this.pendT = pendT;
		}
		
		
		
}
