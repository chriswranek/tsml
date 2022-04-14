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

        //find way of tracking indexes in the array for attributes, atts with values higher than the att size won't work
        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }

        double rootEntropy = 0;

        for (int i = 0; i < classCount.length; i++) {
            rootEntropy += (classCount[i]/numInstances) * log2(classCount[i]/numInstances);
        }

        rootEntropy *= -1;

        for (int i = 0; i < attCount.length; i++) {
            //get class count for each attribute value compared to class count for the whole dataset

            int[] attClassCount = new int[classCount.length];

            for (int[] ints: arr) {
                if(ints[0] == attValues.get(i)){
                    attClassCount[classValues.indexOf(ints[1])]++;
                }
            }

            //Get local entropy for each class value, then sum them to get an entropy for each attribute value
            for (int k : attClassCount) {
                attEntropies[i] += ((double) k / attCount[i]) * log2((double) k / attCount[i]);
                //System.out.println(attEntropies[i]);
            }

            attEntropies[i] *= -1;

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

        //find way of tracking indexes in the array for attributes, atts with values higher than the att size won't work
        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }

        //Calculate impurity at root

        double localGiniIndex = 0;
        for (int i = 0; i < classCount.length; i++) {
            localGiniIndex += (Math.pow(classCount[i] / numInstances, 2));

        }
        double rootGiniIndex = 1 - localGiniIndex;


        for (int i = 0; i < attValues.size(); i++) {

            int[] attClassCount = new int[classCount.length];

            for (int[] ints: arr) {
                if(ints[0] == attValues.get(i)){
                    attClassCount[classValues.indexOf(ints[1])]++;
                }
            }

            for(int k : attClassCount){
                attGinis[i] += Math.pow((double)k / attCount[i], 2);
            }

            attGinis[i] = 1 - attGinis[i];

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
        //Get attribute counts and class counts for each attribute value
        ArrayList<Integer> attValues = new ArrayList<>();
        ArrayList<Integer> classValues = new ArrayList<>();

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
        double numInstances = 0;

        //find way of tracking indexes in the array for attributes, atts with values higher than the att size won't work
        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }

        double[][] classPreds = new double[classCount.length][attCount.length];

        for (int i = 0; i < classCount.length; i++) {
            double classProb = classCount[i] / numInstances;

            for (int j = 0; j < attCount.length; j++) {
                classPreds[i][j] = attCount[j] * classProb;
            }
        }

        double chiSquaredIndex = 0;

        for (int i = 0; i < attCount.length; i++) {

            int[] attClassCount = new int[classCount.length];

            for (int[] ints: arr) {
                if(ints[0] == attValues.get(i)){
                    attClassCount[classValues.indexOf(ints[1])]++;
                }
            }

            for (int j = 0; j < attClassCount.length; j++) {
                double classRatio = Math.pow(attClassCount[j] - classPreds[j][i], 2) / classPreds[j][i];

                chiSquaredIndex += classRatio;
            }
        }

        return chiSquaredIndex;
    }


    public static void main(String[] args) throws Exception {

        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Playgolf_TRAIN.arff";

        Instances test = DatasetLoading.loadData(testDataLocation);

        //System.out.println(" ");
        //System.out.println(" ");

        double a = 0.25;

        double b = 1 - a;

        double rootEnt = -(a * log2(a) + (b) * log2(b));

        IGAttributeSplitMeasure igAtt = new IGAttributeSplitMeasure();

        ChiSquaredAttributeSplitMeasure chiAtt = new ChiSquaredAttributeSplitMeasure();

        GiniAttributeSplitMeasure giniAtt = new GiniAttributeSplitMeasure();

        System.out.println(igAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println(chiAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println(giniAtt.computeAttributeQuality(test, test.attribute(0)));

    }
}
