package VarLenPSO;

import java.util.Random;

import myUtils.WekaDatasetHandle;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

public class SU {

	Instances data;
	float [][] suMatrix;
	public float [] suic;
	int [] relv_fea_idx; //this is only used in this class to reduce the size of the su matrix to store su between the rel_feas only.
	int nbr_rel_fea = 0;

	public SU(Instances dt) {
		//If data is not discretised, discretise it using MDL
		if (dt.attribute(0).isNumeric()) {
			Discretize disTransform = new Discretize();
			disTransform.setUseBetterEncoding(true);
			try {
				disTransform.setInputFormat(dt);
				Instances dis_train = Filter.useFilter(dt, disTransform);
				data = dis_train;
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
			else
				data = dt;
		buildLS();
	}

	public void buildLS () {

		//calculate SU to class and the nbr of relevant features (SU !=0)
		relv_fea_idx = new int[data.numAttributes()-1];
		suic = new float[data.numAttributes()-1];

        for (int i = 0; i < data.numAttributes() - 1; i++) {
                suic[i] = symmetricalUncertainty(i, data.numAttributes()-1);
                if (suic[i] != 0) {
                	relv_fea_idx[i] = nbr_rel_fea;
                	nbr_rel_fea++;
                }
                else
                	relv_fea_idx[i] = data.numAttributes();
        }

    	if(nbr_rel_fea < 5000){
            suMatrix = new float[nbr_rel_fea][];
            for (int f1 = 0, k = 0; f1 < data.numAttributes()-1; f1++) {
            	if (relv_fea_idx[f1] != data.numAttributes()) { //this feature is relevant
            		suMatrix[k] = new float[k];
            		int l = 0; //k is vertical idx, l is horizontal index
                    for (int f2 = 0; f2 < f1; f2++) {
                    	if (relv_fea_idx[f2] != data.numAttributes())  //this feature is relevant
                    		suMatrix[k][l++] = symmetricalUncertainty(f1, f2);
                    }
                    k++;
            	}

            }
        }

    }

	public double get_suic (int i){
		return suic[i];
	}

    public float get_su(int f1,int f2){

    	//Convert i and j into the idx in the su matrix
    	int i = relv_fea_idx[f1], j = relv_fea_idx[f2];
    	if ((i == data.numAttributes()) || (j == data.numAttributes())) { //either of them is irrelevant
    		return (float)1;
    	}
        if( nbr_rel_fea < 5000){
        	try {
        		return i>j ? suMatrix[i][j] : suMatrix[j][i];}
        	catch (Exception e){
//        		System.out.printf("%d, %d => %d, %d ",f1, f2, i, j );
        	}
        }else{
            float r = 0;
            r = symmetricalUncertainty(f1, f2);
            return r;
        }
        return 0;
    }

    float symmetricalUncertainty(int indexOne,int indexTwo){
        float ig,e1,e2;

        ig=informationGain(indexOne,indexTwo);
        e1=entropy(indexOne);
        e2=entropy(indexTwo);

        if((e1+e2) !=(float)0)
            return((float)2 * (ig/(e1+e2)));
        else
            return (float)1;
    }

    float informationGain(int indexOne,int indexTwo){
        return entropy(indexOne) - condEntropy(indexOne,indexTwo);
    }

    float entropy(int attrIndex){
        float ans=0,temp;
        float curIndex [] = new float[data.numInstances()];
        for (int i = 0; i < curIndex.length; i++) {
            curIndex[i] = (float) data.instance(i).value(attrIndex);
        }

        for(int i=0; i< data.attribute(attrIndex).numValues();i++){
            temp=partialProb(attrIndex,i);
            if(temp!=(float)0)
                ans+= temp *(Math.log(temp)/Math.log((float)2.0));
        }
        return -ans;
    }


    float partialProb(int attrIndex,int attrValue){
        int count=0;
        for(int i=0;i<data.numInstances();i++)
            if(data.instance(i).value(attrIndex) == (float)attrValue)
                count++;
        if(count!=0)
            return ((float)count/(float)data.numInstances());
        else
            return (float)0;
    }

    float condEntropy(int indexOne,int indexTwo){
        float ans=0,temp,temp_ans,cond_temp;
        float oneMS [] = new float[data.numInstances()];
        float twoMS [] = new float[data.numInstances()];
        for (int i = 0; i < oneMS.length; i++) {
            oneMS[i] = (float) data.instance(i).value(indexOne);
            twoMS[i] = (float) data.instance(i).value(indexTwo);
        }
        for(int j=0;j<data.attribute(indexTwo).numValues();j++){
            temp=partialProb(indexTwo,j);
            temp_ans=0;

            for(int i=0;i<data.attribute(indexOne).numValues();i++){
                cond_temp=partialCondProb(indexOne,i,indexTwo,j);
                if(cond_temp != (float)0)
                    temp_ans += cond_temp *(Math.log(cond_temp)/Math.log((float)2.0));
            }
            ans+=temp*temp_ans;
        }
        return -ans;
    }

    float partialCondProb(int indexOne,int valueOne,int indexTwo,int valueTwo){
        int num=0,den=0;

        for(int i=0;i<data.numInstances();i++){
            if(data.instance(i).value(indexTwo) == (float)valueTwo){
                den++;
                if(data.instance(i).value(indexOne) == (float)valueOne)
                    num++;
            }
        }

        if(den!=0)
            return (float)num/(float)den;
        else
            return (float)0;
    }

    public static void main(String[] args) throws Exception {
    	Instances data = WekaDatasetHandle.ReadDataset(args[0] + "/data.arff", Integer.parseInt(args[1]));
		WekaDatasetHandle.PrintDatasetCharacter(data);

		int seed = Integer.parseInt(args[1]);
		int fold = Integer.parseInt(args[2]) - 1; //the input fold [1..10] => need to -1
		data.randomize(new Random(1));
		data.stratify(10);
		Instances train = data.trainCV(10, fold);
		Instances test = data.testCV(10, fold);

		SU su = new SU(train);

    }

}
