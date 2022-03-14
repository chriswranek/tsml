package cwranek;

import org.apache.commons.lang3.ArrayUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;


import java.util.Arrays;
import javafx.util.Pair;

public class OneNN extends AbstractClassifier {

    private Instances classifierData;
    private int kNUmber = 1;

    public void setkNUmber(int newKNumber){
        this.kNUmber = newKNumber;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        classifierData = data;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] predProb = new double[classifierData.numClasses()];
        double[] predCounts = new double[classifierData.numClasses()];

        Instance[] neighbours = getNeighbours(instance);

        for (Instance neighbour : neighbours) {
            predCounts[(int) neighbour.classValue()]++;
        }

        for (int i = 0; i < predProb.length; i++) {
            predProb[i] = predCounts[i] / neighbours.length;
        }
        return predProb;
    }


    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] neighborClsVals = new double[kNUmber];

        Instance[] neighbors = getNeighbours(instance);

        for (int i = 0; i < kNUmber; i++) {
            neighborClsVals[i] = neighbors[i].classValue();
        }

        return mostFrequent(neighborClsVals);
    }

    public Instance[] getNeighbours(Instance instance){
        double[] rankings = new double[classifierData.numInstances()];
        Instance[] neighbors = new Instance[kNUmber];

        for (int i = 0; i < classifierData.numInstances(); i++) {
            rankings[i] = distance(instance, classifierData.instance(i));
        }

        for (int j = 0; j < kNUmber; j++) {

            int minIndex = 0;
            double min = rankings[minIndex];

            for (int i = 0; i < rankings.length; i++) {
                if(rankings[i] < min){
                    min = rankings[i];
                    minIndex = i;
                }
            }

            neighbors[j] = classifierData.get(minIndex);
            rankings = removeElement(rankings, minIndex);
        }

        return neighbors;
    }

    double distance(Instance x, Instance y){

        double totalDistance = 0;

        for (int i = 0; i < classifierData.numAttributes() - 1; i++) {
            //totalDistance += Math.abs((x.value(i) - y.value(i))) * Math.abs((x.value(i) - y.value(i)));
            totalDistance += Math.pow(x.value(i) - y.value(i), 2);
        }

        return totalDistance;
    }

    public static double[] removeElement(double[] arr, int index)
    {
        if (arr == null || index < 0
                || index >= arr.length) {
            return arr;
        }

        double[] anotherArray = new double[arr.length - 1];

        for (int i = 0, k = 0; i < arr.length; i++) {

            if (i == index) {
                continue;
            }

            anotherArray[k++] = arr[i];
        }

        return anotherArray;
    }

    static double mostFrequent(double[] arr)
    {
        Arrays.sort(arr);

        double max_count = 1, res = arr[0];
        double curr_count = 1;

        for (int i = 1; i < arr.length; i++)
        {
            if (arr[i] == arr[i - 1])
                curr_count++;
            else
            {
                if (curr_count > max_count)
                {
                    max_count = curr_count;
                    res = arr[i - 1];
                }
                curr_count = 1;
            }
        }

        if (curr_count > max_count)
        {
            max_count = curr_count;
            res = arr[arr.length - 1];
        }

        return res;
    }
}
