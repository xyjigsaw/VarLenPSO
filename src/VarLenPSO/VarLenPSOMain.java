package VarLenPSO;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import clpsofs.VelocityClampBasic;
import fs.Featureselection;
import fs.MyClassifier;
import fs.Problem;
import weka.classifiers.lazy.IB1;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;
import myUtils.RandomBing;
import myUtils.analysis.CPUTime;
import myUtils.analysis.PerformanceResult;
import myUtils.analysis.ResultsV6;
import myUtils.WekaDatasetHandle;
import myUtils.algo.QSort;

/**
 * This class is used to run the compared methods.
 * @author tranbinh
 *
 */
public class VarLenPSOMain {

	private static String[] clsfr_name= {"IB1", "J48", "NB"}; //"3NN","5NN","SVM",
	static int REPLICATE = 10;
	static int [] rankF;
	private static int MAX_ITER = 100;

	private static boolean REINIT = true;
	private static int MAX_ITER_STAG = 7;
	private static int SIZE_DIVISION = 7;

	private static boolean LOCAL_SEARCH = true;
	private static double FLIP_PERCENTAGE = 0.25;
	private static int LS_MAX_ITER = 100;

	/**
	 * Have 5 command line arg: Data/SRBCT_GEMS 1 12 9 100
	 * 0: dataset path: eg. Data/SRBCT_GEMS
	 * 1: PSO run seed: 1
	 * 2: Nbr of Divisions: 12
	 * 3: Nbr of iterations gbest unchanged to apply length changing: 9
	 * 4: Nbr of max iterations to stop: 100
	 *
	 * @throws Exception
	 */

	public static void main(String args[]) throws Exception {
		String[] args1 = {"bank", "1", "12", "9", "100"};
		//1. Read dataset
		Instances data = WekaDatasetHandle.ReadDataset("/Users/reacubeth/Documents/大学/科研/TEC19年9月/VarLenPSO/"+args1[0]+".arff", 1);
		WekaDatasetHandle.PrintDatasetCharacter(data);

		int seed = Integer.parseInt(args1[1]);
		//		int fold = Integer.parseInt(args[2]) - 1; //the input fold [1..10] => need to -1
		data.randomize(new Random(1));
		data.stratify(REPLICATE);

		//Prepare result recording objects
		ResultsV6 result = new ResultsV6("VLPso.txt", REPLICATE, clsfr_name, false);
		result.setDataName(args1[0]);

		//*********************** IMPORTANT PARAMETER SETTING *********************

		REPLICATE = 10;
		MAX_ITER = 100;

		REINIT = true;
		LOCAL_SEARCH = false;
		FLIP_PERCENTAGE = 0.25;
		LS_MAX_ITER = 100;

		//read parameters: size_div and stag_count
		SIZE_DIVISION = Integer.parseInt(args1[2]);
		System.out.printf("Variable length PSO with size division %d \n", SIZE_DIVISION );
		MAX_ITER_STAG = Integer.parseInt(args1[3]);
		System.out.printf("Reinit population when PSO stagnate for %d iterations \n", MAX_ITER_STAG );
		MAX_ITER = Integer.parseInt(args1[4]);
		System.out.printf("Max #Iterations: %d \n", MAX_ITER );

		//*********************** END IMPORTANT PARAMETER SETTING *************

		FileWriter fileWriter = new FileWriter("Run_"+ seed + "_iter_res.txt");
		BufferedWriter iterWriter = new BufferedWriter(fileWriter);
		iterWriter.write("Iter\tGbestFitn\tGbestsize\tAvgFitn\tAvgSize");

		//3.4 For each fold
		for (int fold = 0; fold < REPLICATE; fold++) {
			Instances train = data.trainCV(REPLICATE, fold);
			Instances test = data.testCV(REPLICATE, fold);

			//Rank features
			rankF = RankFeature("SU", train);
			//      			for(int i = 0; i< rankF.length; i++)
			//      				System.out.print(rankF[i] + ", ");

			//Reorder the features in training and test set using the rankF order.
			ArrayList<Integer> fea_idx = new ArrayList<Integer>();
			for(int i = 0; i< rankF.length; i++)
				fea_idx.add(rankF[i]);

			train = WekaDatasetHandle.transformDataset(train, fea_idx, "train");
			test = WekaDatasetHandle.transformDataset(test, fea_idx, "test");


			System.out.println("***********FOLD " + fold);

			VECLPSO(fold, train, test, seed, result, iterWriter);
		}
		iterWriter.close();
		//record the result to file
		result.recordRun(seed);

	}

	private static void VECLPSO(int fold, Instances train, Instances test, long seed, ResultsV6 result, BufferedWriter iterWriter) throws Exception {

		//**********************IMPORTANT SETTINGS
		int number_of_particles =   (train.numAttributes()-1)/20;
		if (number_of_particles > 300)
			number_of_particles = 300;
		if (number_of_particles <= 100)
			number_of_particles = 100;

		int max_number_of_iterations = MAX_ITER;

		System.out.println("Pop size: " + number_of_particles + "\nMax iter: "+ max_number_of_iterations);
		//Velocity
		double w;
		double c = 1.49445;
		//Seed
		RandomBing.Seeder.setSeed(seed);
		//Problem
		Problem problem = new Featureselection();
		problem.setMaxVelocity(0.6);
		problem.setMinVelocity(-0.6);
		//		Problem problem = new localSearchFS(); //actually this only uses LOOCV (instead of 10FCV)
		problem.setClassifier(new IB1());
		problem.setNumFolds(10);
		//		problem.setEvalType("wrapper");
		problem.setDimension(train.numAttributes()-1);


		//Normalise data
		Normalize filter = new Normalize();
		filter.setInputFormat(train);  // initializing the filter once with training set
		train = Filter.useFilter(train, filter);  // configures the Filter based on train instances and returns filtered instances
		test = Filter.useFilter(test, filter);    // create new test set*/


		//Set train fold and test fold to problem object
		problem.setTraining(train);
		problem.setTestSet(test);
		//Swarm
		VLSwarm s = new VLSwarm(FLIP_PERCENTAGE);
		s.setProblem(problem);
		s.setVelocityClamp(new VelocityClampBasic());
		s.setC(c);
		s.prepareLS();

		s.COUNT_LS_FOUND_PBEST = 0;
		//**********************END IMPORTANT SETTINGS

		System.out.println("**********************************************************************************");
		CPUTime a = new CPUTime();
		long startCPUTimeNano = a.getCpuTime();

		s.initialize(SIZE_DIVISION, number_of_particles);

		//5. START PSO
		int iter = 0;
		int nbr_iter_not_improve = 0;
//		int gbest_idx = 0;//, old_gbest_idx = 0, gbest_age = 0;

		//5.2 Update particle fitness and pbest
		boolean local_search = LOCAL_SEARCH; //((iter < 20)  && (iter % 2 == 0)); true
		boolean found_new_gbest = s.updateFitnessAndLSPbest( local_search, LS_MAX_ITER);
		if (found_new_gbest)
			nbr_iter_not_improve = 0;
		else
			nbr_iter_not_improve++;
		s.calculatePc();
		s.renewExemplar();

		while ((iter < max_number_of_iterations) ) { //&& (nbr_iter_not_improve < MAX_ITER_STAG )) {

			int gbest_size = s.getProblem().subsetSize(s.getGbest().getPersonalPosition());
			double avg_fitn = s.averageFitness();
			double avg_size = s.averageSize();
			System.out.printf("iter %d, gbest fitnest: %.2f (%d)| Avg Fit: %.2f, Avg size: %.2f\n", iter, s.getGbest().getPersonalFitness(),
					gbest_size, avg_fitn, avg_size );

			iterWriter.write("\n" +iter + "\t" + s.getGbest().getPersonalFitness() +
					"\t" + gbest_size + "\t" + avg_fitn +"\t"+ avg_size +
					"\t" + (found_new_gbest?"1":"0") );


			if (REINIT && (nbr_iter_not_improve >= MAX_ITER_STAG)) {
				s.reinit();
				s.updateFitnessAndLSPbest( false, LS_MAX_ITER);
				s.calculatePc();
				s.renewExemplar();
				nbr_iter_not_improve = 0;
			}

			iterWriter.write("\t" + s.get_max_size());

			//5.1 Update the inertia w
			w = (0.9 - ((iter / max_number_of_iterations) * 0.5));

			//5.4 Update velocity and position
			s.updateVelocityPosition(w);

			iter++;
			//5.2 Update particle fitness and pbest
			local_search = LOCAL_SEARCH && ((iter < 20)  && (iter % 2 == 0));
			found_new_gbest = s.updateFitnessAndLSPbest( local_search, LS_MAX_ITER);
			if (found_new_gbest)
				{nbr_iter_not_improve = 0; System.out.printf("Gbest changed!\n");}
			else
				nbr_iter_not_improve++;

		}  //end all iterations
		System.out.printf("LS: FOUND/TOTAL: %d/%d = %.2f\n", s.COUNT_LS_FOUND_PBEST,s.TOTAL_LS_CALL, (double)s.COUNT_LS_FOUND_PBEST/s.TOTAL_LS_CALL);


		// ******************** PERFORMANCE EVAL for the fold-th Fold  ***********************
		//3.4. end timer
		long taskCPUTimeNano = a.getCpuTime() - startCPUTimeNano;

		//6. Get the best subset and Transform training and test set by removing unselected features
		int[] selfeatIdx = problem.selFeaIdx(s.getGbest().getPersonalPosition());
		Remove delTransform = new Remove();
		delTransform.setInvertSelection(true);
		delTransform.setAttributeIndicesArray(selfeatIdx);
		delTransform.setInputFormat(problem.getTraining());

		System.out.printf("gbest: subset size / particle size = %d/ %d\n", selfeatIdx.length-1, s.getGbest().getPersonalPosition().size());

		Instances new_train = Filter.useFilter(problem.getTraining(), delTransform);
		Instances new_test = Filter.useFilter(problem.getTestSet(), delTransform);

		Map<String, PerformanceResult> per_result = new HashMap<String, PerformanceResult>();

		for(int i = 0; i< clsfr_name.length; i++) {
			//8. Calculate accuracy
			PerformanceResult res = MyClassifier.CalculateUnbalanceAccuracy(clsfr_name[i],new_train,new_test);
			per_result.put(clsfr_name[i] + " " + fold, res);
			System.out.print(clsfr_name[i] + ": ");
			System.out.println("===> PSO train acc: " + res.getTrain() + ", PSO test acc: " + res.getTest());
		}

		String solution="";
		for(int i = 0; i< selfeatIdx.length-1; i++)	//-1 because the last feature is the class attribute
			solution += rankF[selfeatIdx[i]] +" , ";


		//9. record the train and test result to file
		result.recordFold(fold, iter, solution, s.getGbest().getPersonalFitness(), per_result, new_train.numAttributes()-1, taskCPUTimeNano / 1E6);

	}


	static int[] RankFeature(String rankmethod, Instances trainingset) {
		int[] ranking = new int[trainingset.numAttributes()-1];

		for(int i = 0; i< ranking.length; i++)
			ranking[i] = i;

		switch(rankmethod) {
		case "SU":
			SU su = new SU(trainingset);
			//			for(int i = 0; i< su.suic.length; i++) {
			//				System.out.print(ranking[i] + ":" + su.suic[i] + "\n");
			//			}

			QSort.sort(su.suic, ranking, 0, ranking.length-1);

			//			for(int i = 0; i< su.suic.length; i++) {
			//				System.out.print(ranking[i] + ":" + su.suic[i] + "\n");
			//			}
			break;
		}
		return ranking;
	}

}
