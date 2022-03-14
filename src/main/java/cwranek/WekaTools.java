package cwranek;

import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class WekaTools {

    public WekaTools() throws Exception {
    }

    double accuracy(Classifier c, Instances test) throws Exception {

        int predCount = 0;
        for (Instance i : test){
            double pred = c.classifyInstance(i);
            double actual = i.classValue();

            if(pred == actual){
                predCount++;
            }
        }

        return predCount / (double)test.numInstances();
    }

    Instances loadClassificationData(String filePath) throws IOException {

        Instances train = null;

        try {
            FileReader reader = new FileReader(filePath);
            train = new Instances(reader);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        assert train != null;

        train.setClassIndex(train.numAttributes()-1);

        return train;

    }

    Instances[] splitData(Instances all, double proportion){

        Random rand = new Random();

        Instances[] split = new Instances[2];

        split[0] = new Instances(all);

        split[1] = new Instances(all, 0);

        split[0].randomize(rand);

        //System.out.println(all.numInstances());

        for (int i = 0; i < all.numInstances() * proportion; i++) {

            split[1].add(split[0].instance(i));

        }

        return split;
    }

    double[] classDistribution(Instances data){

        float instances = data.numInstances();

        System.out.println(instances);

        double[] distro = new double[data.numClasses()];

        for (Instance i : data) {
            if(i.value(data.numAttributes()-1) == 0){
                distro[0]++;
            } else if (i.value(data.numAttributes()-1) == 1){
                distro[1]++;
            } else {
                distro[2]++;
            }
        }

        for (int i = 0; i < data.numClasses(); i++) {
            distro[i] = distro[i] / data.numInstances();
        }

        return distro;

    }

    int[][] confusionMatrix(int[] predicted, int[] actual){

        int[][] matrix = new int[2][2];

        for (int i = 0; i < predicted.length-1; i++) {
            if (predicted[i] == actual[i] && predicted[i] == 0){
                matrix[0][0]++;
            } else if (predicted[i] == 0 && actual[i] == 1){
                matrix[0][1]++;
            } else if (predicted[i] == 1 && predicted[i] == actual[i]){
                matrix[1][1]++;
            } else {
                matrix[1][0]++;
            }
        }

        return matrix;

    }

    int[] classifyInstances(Classifier c, Instances test) throws Exception {

        int[] pred = new int[test.numInstances()];
        int count = 0;

        for(Instance i : test){
            pred[count] = (int) c.classifyInstance(i);
            count++;
        }

        return pred;
    }

    int[] getClassValues(Instances data){

        int[] actual = new int[data.numInstances()];
        int count = 0;

        for(Instance i : data){
            actual[count] = (int) i.classValue();
            count++;
        }

        return actual;
    }




}
