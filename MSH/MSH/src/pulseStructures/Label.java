package pulseStructures;

/**
 * This class represents a label. Inside each label we store the cost and resource attributes.
 * @author nick0
 *
 */
public class Label {

	/**
	 * Attributes of a label: Cost, Time, Walking time
	 */
	public int[] attributes;

	public Label(int[] n_attris) {
		attributes = n_attris.clone();
	}
	/**
	 * True if this dominates param
	 * @param newLabel_attributes
	 * @return
	 */
	public boolean dominateLabel(int[] newLabel_attributes) {
		for (int i = 0; i < newLabel_attributes.length; i++) {
			if(attributes[i]>newLabel_attributes[i]){
				return false;
			}
		}
		return true;
	}

	@Override
	public String toString() {
		return "";
	}
}
