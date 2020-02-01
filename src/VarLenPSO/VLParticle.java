package VarLenPSO;


public class VLParticle extends clpsofs.CLParticle {
	int idx_last_selected_feature = 0; //used for reinit the particle

	public int getIdx_last_selected_feature() {
		return idx_last_selected_feature;
	}

	public void setIdx_last_selected_feature(int idx_last_selected_feature) {
		this.idx_last_selected_feature = idx_last_selected_feature;
	}

	public void remove_pos(int i) {
		_position.remove(i);
		_velocity.remove(i);
		_personal_position.remove(i);
		_exemplar.remove(i);

	}

	public void add_pos() {
		_position.add(0.0);
		_velocity.add(0.0);
		_personal_position.add(0.0);
		_exemplar.add(0);

	}
}
