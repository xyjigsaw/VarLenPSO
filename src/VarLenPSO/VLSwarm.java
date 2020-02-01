package VarLenPSO;

import clpsofs.Initialisation;
import eclpsofs.Swarm;
import myUtils.algo.QSort;
import myUtils.analysis.SmallStatistic;
import myUtils.maths.NewMath;

public class VLSwarm extends Swarm {

	//For local search:
	protected double _percentFlip;
	public SU _su;
	public int COUNT_LS_FOUND_PBEST = 0;
	public int TOTAL_LS_CALL = 0;
	//to divide particles into different sizes
	protected int _size_division = 0;
	protected int _max_size = 0;

	public VLSwarm(double percent_flip){
		_percentFlip = percent_flip;
//		System.out.printf("Local Search Pbest Swarm with %.2f%% flip \n",percent_flip);
	}

	public void prepareLS(){
		_su = new SU(getProblem().getTraining());
	}

	@Override
	public void renewExemplar(int p) {
		VLParticle par = (VLParticle) getParticle(p);
		//		System.out.println("\nRenew Exemplar for " + p);
		for (int d = 0; d < par.getSize(); ++d) {
			//generate a random nbr
			double rnd = _random.nextDouble();
			int exempl = p;

			if (rnd >= _Pc[p])
				exempl = p;
			else {
				int attemp = 0;
				//use tournament to select one particle as exemplar, make sure p1 and p2 and p are different particles
				int p1 = (int)Math.round(_random.nextDouble() * (numberOfParticles()-1));
				boolean satisfy = false;
				while (!satisfy && attemp < numberOfParticles()) {
					p1 = (int)Math.round(_random.nextDouble() * (numberOfParticles()-1));
					attemp++;
					satisfy = ((p1 != p) && (getParticle(p1).getSize() > d));
				}
				if (!satisfy) p1 = p;

				int p2 = (int)Math.round(_random.nextDouble() * (numberOfParticles()-1));
				attemp = 0;
				satisfy = false;
				while (!satisfy && attemp < numberOfParticles()) {
					p2 = (int)Math.round(_random.nextDouble() * (numberOfParticles()-1));
					attemp++;
					satisfy = ((p2 != p) && (p2 != p1) && (getParticle(p2).getSize() > d));
				}
				if (!satisfy) p2 = p;

				if(getParticle(p1).getPersonalFitness() > getParticle(p2).getPersonalFitness() )
					exempl = p1;
				else
					exempl = p2;
			}
			getParticle(p).setExemplar(d, exempl);
		}
	}

	public void initParticle(VLParticle par, int type) {

		double[] position = new double[par.getSize()];
		if (type == 0) //normal
			position = Initialisation.NormalInitialisation(par.getSize(), getProblem());
		else //type ==1 //all position is 1
			for (int d = 0; d < par.getSize(); ++d) {
				position[d] = 1.0;
			}

		for (int d = 0; d < par.getSize(); ++d) {
			par.setPosition(d, position[d]);
			par.setPersonalPosition(d, position[d]);

			//				double velocity = NewMath.Scale(0, 1, RandomBing.Create().nextDouble(), getProblem().getMinVelocity(), getProblem().getMaxVelocity());
			double velocity = NewMath.Scale(0, 1, _random.nextDouble(), getProblem().getMinVelocity(), getProblem().getMaxVelocity());
			//                System.out.printf("%.2f, " ,velocity);
			//				if (velocity < 0)
			//					neg_velocity_count++;
			par.setVelocity(d, velocity);
			par.setPersonalFitness(getProblem().getWorstFitness());

			//				par.setExemplar(d, p);
		}
	}

	public void initialize(int size_division, int number_of_particles) {
		_size_division = size_division;

		_max_size = getProblem().getDimension();
		for (int i = 0; i < _size_division; ++i) {
			VLParticle p = new VLParticle();
			p.setSize((i+1) * getProblem().getDimension()/_size_division);
			addParticle(p);
			initParticle(p, 1); //add one particle with all 1s in position (i.e. select all features)
			for(int j = 1; j < number_of_particles/ _size_division; ++j) {
				p = new VLParticle();
				p.setSize((i+1) * getProblem().getDimension()/_size_division);
				addParticle(p);
				initParticle(p, 0);
			}
		}

		_gbest = new VLParticle();
		_gbest.copyParticle(getParticle(0));
		//Init Pc array
		_Pc = new double[numberOfParticles()];
		refresh_gap_count = new int[numberOfParticles()];

	}

	public boolean updateFitnessAndLSPbest(boolean local_search, int LS_max_times) {

		boolean have_new_gbest = false;
		for (int p = 0; p < numberOfParticles(); ++p) {
			VLParticle par_i = (VLParticle) getParticle(p);

			if (getProblem().subsetSize(par_i.getPosition()) == 0 )
				par_i.setFitness(getProblem().getWorstFitness());
			else
			{
				double new_fitness = getProblem().fitness(par_i.getPosition());
				par_i.setFitness(new_fitness);

				//Check if new position is better than personal position...
				double is_better = getProblem().compare(par_i.getFitness(), par_i.getPersonalFitness());
				if (( is_better > 0) ||
						( (is_better == 0 ) && //equal fitness
								(getProblem().subsetSize(par_i.getPosition())
										< getProblem().subsetSize(par_i.getPersonalPosition()) )) )  //smaller size
				{ //update pbest
					par_i.setPersonalFitness(par_i.getFitness());
					for (int j = 0; j < par_i.getSize(); ++j) {
						par_i.setPersonalPosition(j, par_i.getPosition(j));
					}

					if (local_search) {
						TOTAL_LS_CALL ++;
						KNNopt myknn = new KNNopt(1, getProblem().getTraining(), 123);

						boolean[] selected_pbest = getProblem().positionToBinarySubset(par_i.getPersonalPosition());
						//                     int flip_size = (int) Math.round(getProblem().subsetSize(p_i.getPosition()) * percentFlip);

						SolutionPack new_pbest = myknn.localsearchSU(_su, selected_pbest,
								par_i.getPersonalFitness(), _percentFlip, LS_max_times, true, getProblem());

						//check to copy new solution to pbest
						if (new_pbest != null) {
							COUNT_LS_FOUND_PBEST ++;
							par_i.setPersonalFitness(new_pbest.acc);
							par_i.setIdx_last_selected_feature(0);

							for (int j = 0; j < par_i.getSize(); ++j) {
								if (new_pbest.sol[j]) {
									par_i.setPersonalPosition(j, 1);
									if (par_i.getIdx_last_selected_feature() < j)
										par_i.setIdx_last_selected_feature(j);
								} else {
									par_i.setPersonalPosition(j, 0);
								}
							}


						}
					}
					//check to update gbest
					is_better = getProblem().compare(par_i.getPersonalFitness(), _gbest.getPersonalFitness());
					if (( is_better > 0) ||
							( (is_better == 0 ) && //equal fitness
									(getProblem().subsetSize(par_i.getPosition())
											< getProblem().subsetSize(_gbest.getPersonalPosition()) )) ) {
						_gbest.copyParticle(par_i);
						have_new_gbest = true;
					}

					refresh_gap_count[p] = 0;
				}
				else {//no improvement
					refresh_gap_count[p] ++;
					if (refresh_gap_count[p] == REFRESH_GAP)
						REASSIGN = true;
				}
			}
		}// end all particles
		return have_new_gbest;
	}

	public void reinit() {

		//Find the size that has the best avg pbest fitness.
		int [] size = new int[_size_division];
		SmallStatistic[] avg_fit = new SmallStatistic[_size_division];
		for (int i = 0; i < _size_division; ++i) {
			int start_particle_idx = i * _swarm.size()/_size_division;
			size[i] = getParticle(start_particle_idx).getSize();
			avg_fit[i] = new SmallStatistic();
			for(int j = 0; j < _swarm.size()/_size_division; ++j) {
				if (getParticle(start_particle_idx + j).getSize() != size[i])
					System.out.println("Uh Ohhh: wrong code");
				avg_fit[i].add(getParticle(start_particle_idx + j).getPersonalFitness());
			}
		}
		int best_idx = 0; double best_avg_fit = 0;
		for(int i = 0; i< avg_fit.length; i++) {
			if (best_avg_fit < avg_fit[i].getAverage()) {
				best_avg_fit = avg_fit[i].getAverage();
				best_idx = i;
			}
		}

		System.out.print("\nSize: ");
		for(int j = 0; j< size.length; j++) {
			System.out.print(size[j]+ ", ");
		}
		System.out.println("Best is " + size[best_idx]);

		if (_max_size != size[best_idx]) {
			_max_size = size[best_idx];
			ResizeParticles(size);
		}
		else {
			System.out.println("Best size unchanged");
		}
	}

	public int get_max_size() {
		return _max_size;
	}

	/**
	 * This is the changing length used in VarLenPSO paper
	 * @param size
	 */
	void ResizeParticles(int[] size) {
		System.out.println("Resize all particles to the best size " + _max_size);
		for (int i = 0, k = 1; i < _size_division; ++i) {
			if (size[i] != _max_size) {
				int new_size = k * _max_size / _size_division;
				for(int j = 0; j < _swarm.size()/_size_division; ++j) {
					VLParticle par_i = (VLParticle) getParticle(i * _swarm.size()/_size_division + j);
					int cur_size = par_i.getSize();
					if ( cur_size > new_size) //remove the last positions
						for (int d = cur_size-1; d>= new_size; d--) {
							par_i.remove_pos(d);
						}
					else if (cur_size < new_size)  //append more positions
						for (int d = cur_size; d < new_size; d++) {//remove the last positions
							par_i.add_pos();
							par_i.setPosition(d, _random.nextDouble());
							par_i.setVelocity(d, NewMath.Scale(0, 1, _random.nextDouble(), getProblem().getMinVelocity(), getProblem().getMaxVelocity()));
						}
					//initParticle(par_i, 0);
				}
				k++;
			}

		}
	}
}
