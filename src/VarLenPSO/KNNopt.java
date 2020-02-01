/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package VarLenPSO;

import edu.princeton.cs.algs4.IndexMinPQ;
import fs.MyClassifier;
import fs.Problem;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Random;

import myUtils.algo.QSort;
import weka.core.Instances;


/**
 *
 * @author
 */
public class KNNopt {
	int K = 3;
	protected Instances dat;
	protected Random rnd;
	protected double[][] insDistance;

	public static void main(String[] args) throws IOException {
		/*        MyDiscDataset dat = new MyDiscDataset("dataset/11_Tumors_GEMS.txt", " ",0);
//        dat.normalise();
         KNNopt myknn = new KNNopt(1, dat, 123);
        for (int iter = 1; iter < 10; iter++) {
            boolean[] selected = myknn.randomSelectedFeature();
            //double acc = myknn.localsearch(selected, myknn.getAccurarySelectedFeatures(selected), 100, 100, false).acc; //myknn.KFoldcrossvalidation(dat.nInstance, selected);
            //System.out.println("++++++++++++++++++++++++ \naccuracy ("  + myknn.K + ") = " + acc);
        }*/
	}

	/**
	 * Constructor of KNNopt
	 * @param k K-neighbours
	 * @param d training dataset
	 */
	public KNNopt(int k, Instances d, int seed) {
		rnd = new Random(seed);
		K = k;
		dat = d;
	}

	/**
	 * Prepare cross distances for selected feature subsets:
	 * Calculate sum distances of all cross instances in the dataset	 *
	 * @param selected
	 * @return the distance matrix
	 */
	protected double[][] preprocessing(Instances data, boolean[] selected) {

		double[][] distances;
		distances = new double[data.numInstances()][data.numInstances()];
		for (int i = 0; i < data.numInstances()-1; i++) {
			for (int j = i+1; j < data.numInstances(); j++) {
				double distance = 0;
				for (int k = 0; k < selected.length; k++) {
					if(selected[k]) { // && (data.instance(j).value(k) != data.instance(i).value(k))){
						double diff = data.instance(j).value(k) - data.instance(i).value(k);
						//Using Euclidean distance
//						distance += diff * diff;
						//Using Manhattan distance
						distance += Math.abs(diff);
						//Overlapping in discrete data
//						distance += ( data.instance(i).value(k) == data.instance(j).value(k) ? 0 : 1);

					}
				}
				distances[j][i] = distances[i][j] = distance;
			}
		}
		return distances;
	}

	/**
	 * Using different distance: Euclidean, Manhattan, or Overlapping
	 * @param selectedFeatures
	 * @param flipfeature
	 * @param distances
	 * @param curSize
	 * @return
	 */
	public double DistanceAndLOOCVAcc(boolean[] selectedFeatures, int[] flipfeature, double[][] distances , int curSize, int CurMaxDiff) {

//		data.randomize(new Random(1));
//		data.stratify(data.numInstances()); //to make it same as the performance eval at the end.
		double[][] confusion_matrix = new double[dat.numClasses()][dat.numClasses()];

		//1. update distance between pair of instances based on flipped indexes
		int nbr_inst = dat.numInstances();
		double d_b = 0, d_w = 0;
		for (int i = 0; i < nbr_inst; i++) {
			double min_d_b = 1, max_d_w = 0;
			int nearest_ins = 0;
			double min_distance = Double.MAX_VALUE;

			//Using Manhattan distance
			for (int j = 0; j < nbr_inst; j++) {
				if(i!=j) {
				//get the old distance between instance i and j
				double dis = distances[i][j];
				int tmp_size = curSize;
				if (flipfeature!=null) {
					for (int k = 0; k < flipfeature.length; k++) {
						double diff = dat.instance(i).value(flipfeature[k]) - dat.instance(j).value(flipfeature[k]);
							if (selectedFeatures[flipfeature[k]] ) {
								dis -= Math.abs(diff);
								tmp_size --;
							}
							else {
								dis += Math.abs(diff);
								tmp_size ++;
							}
					}
				} //end update dis
				double nor_dis = dis / tmp_size;
				//******** End Manhattan distance

				if(dat.instance(i).classValue() == dat.instance(j).classValue()) {//same class => update max_d_w
					if ( nor_dis > max_d_w)
						max_d_w = nor_dis;
				}
				else { //different class => update min_d_b
					if (nor_dis < min_d_b)
						min_d_b = nor_dis;
				}
				if (dis < min_distance ){
					min_distance = dis;
					nearest_ins = j;
				}
				} //end if i!= j
			}//end for instance j
			d_b += min_d_b;
			d_w += max_d_w;
			confusion_matrix[(int)dat.instance(i).classValue()][(int)dat.instance(nearest_ins).classValue()] += 1;
		}//end for instance i
		d_b /= nbr_inst;
		d_w /= nbr_inst;

		double dist = 1.0 / (1.0 + Math.exp(-5.0 * (d_b - d_w)));
		return 0.8 * MyClassifier.unbalanceAcc(confusion_matrix) + 0.2 * dist;

	}

	/**
	 * generate 2 random feature subset which has different % of 0s and 1s.
	 * @return feature set
	 */
	protected void halfOnesZeros(boolean[] sol, int [] ones, int[] zeros) {

		int nbr_0 = zeros.length;
		int nbr_1 = ones.length;
		int zero_idx = 0, one_idx = 0;
		boolean do_it_again;
		do {
			do_it_again = false;
			IndexMinPQ sort = new IndexMinPQ(sol.length);
			for (int i = 0; i < sol.length; i++) {
				sort.insert(i, rnd.nextDouble());
			}

			try {
				while ((zero_idx < nbr_0) && (one_idx < nbr_1)) {
					int tmp = sort.delMin();
					if (sol[tmp]) {
						ones[one_idx] = tmp;
						one_idx++;
					} else {
						zeros[zero_idx] = tmp;
						zero_idx++;
					}
				}
				while (one_idx < nbr_1) {
					int tmp = sort.delMin();
					if (sol[tmp]) {
						ones[one_idx] = tmp;
						one_idx++;
					}
				}
				while ((zero_idx < nbr_0)) {
					int tmp = sort.delMin();
					if (!(sol[tmp])) {
						zeros[zero_idx] = tmp;
						zero_idx++;
					}
				}
			} catch (NoSuchElementException e) {
				do_it_again = true;
			}

		} while (do_it_again);

	}



	/**
	 *
	 * @param oldDistance
	 * @param oldsolution
	 * @param flip
	 * @return
	 */
	protected int updatingdistance(double[][] oldDistance,
			boolean[] oldsolution, int[] flip, int cur_max_diff) {

		int ret_max_diff = cur_max_diff;
		for (int i = 0; i < dat.numInstances()-1; i++) {
			for (int j = i+1; j < dat.numInstances(); j++) {
				for (int k = 0; k < flip.length; k++) {

					double diff = dat.instance(i).value(flip[k]) - dat.instance(j).value(flip[k]);
					if (oldsolution[flip[k]]) { //if feature at flip[k] was selected (true) in old solution => subtract its dis. from sum dis.
						//Using Euclidean distance
//						oldDistance[i][j] -= diff * diff;
						//Using Manhattan distance
						oldDistance[i][j] -= Math.abs(diff);
						//Using Overlapping distance
//						oldDistance[i][j] -= (dat.instance(i).value(flip[k]) == dat.instance(j).value(flip[k]) ? 0 : 1);
					}//else = feature at flip[k] was not selected (false) in old solution => add its dis. to sum dis.
					else {
						//Using Euclidean distance
//						oldDistance[i][j] += diff * diff;
						//Using Manhattan distance
						oldDistance[i][j] += Math.abs(diff);
						//Using Overlapping distance
//						oldDistance[i][j] += (dat.instance(i).value(flip[k]) == dat.instance(j).value(flip[k]) ? 0 : 1);
					}
				}
				oldDistance[j][i] = oldDistance[i][j];
			}
		}
		//update max_diff
		for (int k = 0; k < flip.length; k++) {
			if (oldsolution[flip[k]])
				ret_max_diff -= dat.attribute(flip[k]).numValues();
			else
				ret_max_diff += dat.attribute(flip[k]).numValues();

		}
		return ret_max_diff;

	}

	protected int[] flip(SU su, int[] ones, int[] zeros) {
		int[] feature_to_flip = new int[ones.length + zeros.length];

		//get su of ones
		double [] su_ones = new double[ones.length];
		double avg_su = 0;
		for (int x = 0; x< ones.length; x++){
			su_ones[x] = su.get_suic(ones[x]);
			avg_su += su_ones[x];
		}
		avg_su /= ones.length;
		QSort.sort(su_ones, ones, 0, ones.length-1);

		//consider the ones
		int k = 0;
		for(int i= 0; i < ones.length-1; i++)
			if (ones[i] != Integer.MAX_VALUE) {
				for (int j = i+1; j< ones.length; j++)
					if (ones[j] != Integer.MAX_VALUE){
						double su_btw_fea = su.get_su(ones[i],ones[j]);
						if ( su_btw_fea > su.get_suic(ones[j]) ) { // flip j off
							feature_to_flip[k++] = ones[j];
							ones[j] = Integer.MAX_VALUE;
						}
					}

			}

		//consider the zeros
		for (int x = 0; x < zeros.length; x++)
			if ( su.get_suic(zeros[x]) > avg_su)
				feature_to_flip[k++] = zeros[x];

		//resize the feature_to_flip to k features
		if (k > 0) {
			int[] f_to_flip = new int [k];
			System.arraycopy(feature_to_flip, 0, f_to_flip, 0, k);
			return f_to_flip;}
		else return null;
	}

	public SolutionPack localsearchSU(SU su, boolean[] sol, double initialAcc, double flipPercent, int maxEval, boolean needPreprocessing, Problem prb)  {
		// iterate through all neigbours
		double best_fitn = initialAcc;
		int best_size = prb.booleanSubsetSize(sol);
		int cur_max_diff = 0;
		boolean found = false;

//		System.out.printf("\n****************Local search START ***************:\n");// init acc = %.4f, size: %d ",initialAcc, prb.booleanSubsetSize(sol));
		//Calculate all cross distances for each data instance
		if (needPreprocessing) {

			insDistance = preprocessing(dat, sol);

		}

		int[] best_flip_pos = null;
		for( int count = 0; count < maxEval; count++) {
			int flipSize = (int) (flipPercent * prb.booleanSubsetSize(sol));
			int zero_size = (int) Math.round( rnd.nextDouble() * (sol.length - best_size > flipSize ? flipSize : (sol.length - best_size)));
			int[] zeros = new int[zero_size];
			int[] ones = new int[flipSize - zero_size];
			int[] feature_to_flip = null;
			int tries = 0;
			while (feature_to_flip == null && tries <10) {
				halfOnesZeros(sol, ones, zeros);
				feature_to_flip = flip(su, ones, zeros);
				tries++;
			}

			if(feature_to_flip != null){
				double fitn =
						DistanceAndLOOCVAcc(sol, feature_to_flip, insDistance, best_size, cur_max_diff);
//						LOOCV(sol, feature_to_flip, insDistance);

				//Calculate the new solution
				boolean[] sol_tmp = new boolean[sol.length];
				sol_tmp = sol.clone();
				for (int i = 0; i < feature_to_flip.length; i++) {
					sol_tmp[feature_to_flip[i]] = !sol_tmp[feature_to_flip[i]];
				}
				int tmp_size = prb.booleanSubsetSize(sol_tmp);
				if ( prb.isBetter(fitn, best_fitn) ||
						((fitn == best_fitn) &&  tmp_size < best_size) )
				{
					best_fitn = fitn;
					best_flip_pos = feature_to_flip.clone();
					best_size = tmp_size;

					//Update distance matrix according to new solution and cur_max_diff
					cur_max_diff = updatingdistance(insDistance, sol, best_flip_pos, cur_max_diff);
					//Update the solution
					for (int i = 0; i < best_flip_pos.length; i++) {
						sol[best_flip_pos[i]] = !sol[best_flip_pos[i]];
					}
					found = true;
				}
			}//end if
		}//end for count

		if (found) {
//			System.out.printf("-> LS found acc = %.4f, size: %d ",best_acc,best_size);
			return new SolutionPack(sol, best_fitn);
		}
		else {
//			System.out.printf("\n-> LS NOT found \n");
			return null;
		}

	}


}

