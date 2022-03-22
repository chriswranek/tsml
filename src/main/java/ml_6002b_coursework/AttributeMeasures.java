package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Empty class for PArt 2.1 of the coursework
 *
 */
public class AttributeMeasures {

    public static double log2(double x){
        return (Math.log(x) / Math.log(2));
    }

    public static double measureInformationGain(int[][] arr){

        ArrayList<Integer> attValues = new ArrayList<>();
        ArrayList<Integer> classValues = new ArrayList<>();
        ArrayList<Integer> attWins = new ArrayList<>();

        for (int[] ints : arr) {
            if (!attValues.contains(ints[0])) {
                attValues.add(ints[0]);
            }
            if (!classValues.contains(ints[1])) {
                classValues.add(ints[1]);
            }
        }

        int[] attCount = new int[attValues.size()];
        int[] classCount = new int[classValues.size()];
        double[] attEntropies = new double[attCount.length];
        double numInstances = 0;

        for (int[] ints: arr) {
            attCount[ints[0]]++;
            classCount[ints[1]]++;
            numInstances++;
        }

        double rootEntropy = -((classCount[1]/numInstances) * log2(classCount[1]/numInstances) + (classCount[0]/numInstances) * log2(classCount[0]/numInstances));

        for (int i: attValues) {
            int winCount = 0;
            for (int[] ints: arr) {
                if(ints[0] == i && ints[1] == 1){
                    winCount++;
                }
            }
            attWins.add(winCount);
        }

        for (int i = 0; i < attValues.size(); i++) {
            attEntropies[i] = -(((double)attWins.get(i) / (double)attCount[i]) * log2((double)attWins.get(i) / (double)attCount[i]) +
                    (((double)attCount[i] - (double)attWins.get(i)) / (double)attCount[i]) * log2((attCount[i] - (double)attWins.get(i)) / (double)attCount[i]));

            if(Double.isNaN(attEntropies[i])){
                attEntropies[i] = 0;
            }
        }

        double attGain = 0.0;
        for (int i = 0; i < attValues.size(); i++) {
            attGain += (attCount[i] / numInstances) * attEntropies[i];
        }

        return rootEntropy - attGain;
    }

    public static double measureInformationGainRatio(int[][] arr){

        double numInstances = 0;
        int winCount = 0;
        int lossCount = 0;

        for (int[] ints: arr) {
            if(ints[1] == 1){
                winCount++;
            } else {
                lossCount++;
            }
            numInstances++;
        }

        double intrinsicValue = -(winCount/numInstances * log2(winCount/numInstances) + lossCount/numInstances * log2(lossCount/numInstances));

        return measureInformationGain(arr) / intrinsicValue;
    }

    public static double measureGini(int[][] arr){
        //Get attribute counts and class counts for each attribute value
        ArrayList<Integer> attValues = new ArrayList<>();
        ArrayList<Integer> classValues = new ArrayList<>();
        ArrayList<Integer> attWins = new ArrayList<>();

        for (int[] ints : arr) {
            if (!attValues.contains(ints[0])) {
                attValues.add(ints[0]);
            }
            if (!classValues.contains(ints[1])) {
                classValues.add(ints[1]);
            }
        }

        int[] attCount = new int[attValues.size()];
        int[] classCount = new int[classValues.size()];
        double[] attGinis = new double[attCount.length];
        double numInstances = 0;

        for (int[] ints: arr) {
            attCount[ints[0]]++;
            classCount[ints[1]]++;
            numInstances++;
        }

        for (int i: attValues) {
            int winCount = 0;
            for (int[] ints: arr) {
                if(ints[0] == i && ints[1] == 1){
                    winCount++;
                }
            }
            attWins.add(winCount);
        }

        //Calculate impurity at root.
        double rootGiniIndex = 1 - (Math.pow(classCount[1] / numInstances, 2) + Math.pow(classCount[0] / numInstances, 2));

        for (int i = 0; i < attValues.size(); i++) {

            attGinis[i] = 1 - (Math.pow((double)attWins.get(i) / attCount[i], 2) + Math.pow(((double)attCount[i] - attWins.get(i)) / attCount[i], 2));

            if(Double.isNaN(attGinis[i])){
                attGinis[i] = 0;
            }
        }

        for (int i = 0; i < attValues.size(); i++) {
            rootGiniIndex -= (attCount[i] / numInstances) * attGinis[i];
        }

        return rootGiniIndex;
    }

    public static double measureChiSquared(int[][] arr){
        ArrayList<Integer> attValues = new ArrayList<>();
        ArrayList<Integer> classValues = new ArrayList<>();
        ArrayList<Integer> attWins = new ArrayList<>();
        ArrayList<Integer> attLosses = new ArrayList<>();

        for (int[] ints : arr) {
            if (!attValues.contains(ints[0])) {
                attValues.add(ints[0]);
            }
            if (!classValues.contains(ints[1])) {
                classValues.add(ints[1]);
            }
        }

        int[] attCount = new int[attValues.size()];
        int[] classCount = new int[classValues.size()];
        double[] attGinis = new double[attCount.length];

        double numInstances = 0;

        for (int[] ints: arr) {
            attCount[ints[0]]++;
            classCount[ints[1]]++;
            numInstances++;
        }

        for (int i: attValues) {
            int winCount = 0;
            int lossCount = 0;
            for (int[] ints: arr) {
                if(ints[0] == i && ints[1] == 1){
                    winCount++;
                } else if (ints[0] == i && ints[1] == 0){
                    lossCount++;
                }
            }
            attWins.add(winCount);
            attLosses.add(lossCount);
        }

        double[] predWins = new double[attCount.length];
        double[] predLoss = new double[attCount.length];

        double winProb = classCount[1] / numInstances;
        double lossProb = classCount[0] / numInstances;

        for (int i = 0; i < attCount.length; i++) {
            predWins[i] = attCount[i] * winProb;
            predLoss[i] = attCount[i] * lossProb;
        }

        double chiSquaredIndex = 0;

        for (int i = 0; i < attCount.length; i++) {
            double winRatio = Math.pow(attWins.get(i) - predWins[i], 2) / predWins[i];

            double lossRatio = Math.pow(attLosses.get(i) - predLoss[i], 2) / predLoss[i];

            chiSquaredIndex += winRatio + lossRatio;
        }

        return chiSquaredIndex;
    }


    public static void main(String[] args) throws Exception {

        int[][] outlookTest = {{0, 0}, {0, 0}, {0, 0}, {0, 1}, {0, 1}};

        int[][] chiSquaredTest = {{2, 3, 5}, {4, 0, 4}, {3, 2, 5}, {9, 5, 14}};

        int[][] peatyTest = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Whisky_TRAIN.arff";

        Instances test = DatasetLoading.loadData(testDataLocation);

        //System.out.println("Information gain for Test Data = " + measureInformationGain(outlookTest));
        //System.out.println("Information gain ratio for Test Data = " + measureInformationGainRatio(outlookTest));

        //System.out.println(" ");
        //System.out.println(" ");

        //System.out.println(measureGini(peatyTest));

        //System.out.println(measureChiSquared(peatyTest));

        double a = 0.25;

        double b = 1 - a;

        //System.out.println(a);

        //System.out.println(b);

        double rootEnt = -(a * log2(a) + (b) * log2(b));


        //System.out.println(rootEnt);

        IGAttributeSplitMeasure igAtt = new IGAttributeSplitMeasure();

        ChiSquaredAttributeSplitMeasure chiAtt = new ChiSquaredAttributeSplitMeasure();

        GiniAttributeSplitMeasure giniAtt = new GiniAttributeSplitMeasure();

        System.out.println(igAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println(chiAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println(giniAtt.computeAttributeQuality(test, test.attribute(0)));

    }
}
