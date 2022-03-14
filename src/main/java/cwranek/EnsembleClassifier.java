package cwranek;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class EnsembleClassifier extends AbstractClassifier {

    private int m_numOfClassifiers = 10;
    protected int randomSeed = 1;
    private J48[] j48arr;

    public void setM_numOfClassifiers(int numOfClassifiers){
        this.m_numOfClassifiers = numOfClassifiers;
    }

    public int getM_numOfClassifiers(){
        return m_numOfClassifiers;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        j48arr = new J48[m_numOfClassifiers];

        WekaTools tools = new WekaTools();

        for (int i = 0; i < m_numOfClassifiers; i++) {
            j48arr[i] = new J48();

            Instances[] trainTest;
            //j48 = new J48();
            Random rand = data.getRandomNumberGenerator(randomSeed);
            data.randomize(rand);
            trainTest = tools.splitData(data, 0.5);

            j48arr[i].buildClassifier(trainTest[1]);
        }

    }

    public double classifyInstance(Instance instance) throws Exception {

        int[] classCounts = new int[j48arr.length];

        double maxNum = classCounts[0];

        for (int i = 0; i < j48arr.length - 1; i++) {
            double pred = j48arr[i].classifyInstance(instance);
            //double actual = instance.classValue();
            classCounts[i] += 1;

        }

        for (int j : classCounts) {
            if(j > maxNum){
                maxNum = j;
            }
        }

        return maxNum;
    }


}
