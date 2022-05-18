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

        //First create and populate arraylists with the all the different attribute values
        //and class values in the dataset. These arraylists are used for indexing the attribute values later
        for (int[] ints : arr) {
            if (!attValues.contains(ints[0])) {
                attValues.add(ints[0]);
            }
            if (!classValues.contains(ints[1])) {
                classValues.add(ints[1]);
            }
        }

        //Create integer arrays to store the counts of each attribute value and class value in the dataset.
        int[] attCount = new int[attValues.size()];
        int[] classCount = new int[classValues.size()];


        //Create an array to store the entropy values for each different attribute value
        double[] attEntropies = new double[attCount.length];
        double numInstances = 0;

        //Iterates through the array, incrementing the count for each attribute value and class value
        //These count arrays are used to measure the probabilities of class outcomes based on attribute values later
        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }


        double rootEntropy = 0;
        //Calculates the root entropies by taking each class probability and multiplying it by the log base 2 of itself.
        for (int i = 0; i < classCount.length; i++) {
            rootEntropy += (classCount[i]/numInstances) * log2(classCount[i]/numInstances);
        }

        //The root entropy must be a negative value so its multiplied by -1
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
            }

            attEntropies[i] *= -1;

            //If the entropy value returns as NaN, then it is set to 0.
            if(Double.isNaN(attEntropies[i])){
                attEntropies[i] = 0;
            }
        }

        //Calculates the information gain for the attribute by taking the probability of the attribute occuring and
        //multiplying it by the attributes entropy value, this is done for each attribute value and is then summed to give
        //an overall gain for attribute as a whole
        double attGain = 0.0;
        for (int i = 0; i < attValues.size(); i++) {
            attGain += (attCount[i] / numInstances) * attEntropies[i];
        }

        //The attribute entropy is subtracted from the root entropy to give the final information gain for the attribute
        //and the value is returned.
        return rootEntropy - attGain;
    }

    public static double measureInformationGainRatio(int[][] arr){

        double numInstances = 0;

        ArrayList<Integer> classValues = new ArrayList<>();

        //First create and populate arraylist with the all the different class values in the dataset.
        // This arraylist is used for indexing the class values later
        for (int[] ints: arr) {
            if (!classValues.contains(ints[1])) {
                classValues.add(ints[1]);
            }
        }

        //Create integer arrays to store the counts of each class value in the dataset.
        int[] classCount = new int[classValues.size()];

        //Iterates through the array, incrementing the count for each class value
        //This count array is used to calculate the datas intrinsic value later
        for (int[] ints: arr) {
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }

        double intrinsicValue = 0;

        //calculates the intrinsic value for the data by iterating through each class value, using its count divided by
        //the number of instances to get an intrinsic probability for each class being predicted in the dataset.
        for (int j : classCount) {
            intrinsicValue += (j / numInstances * log2(j / numInstances));
        }

        intrinsicValue *= -1;

        //The ratio is simply the regular information gain divided by the intrinsic value
        return measureInformationGain(arr) / intrinsicValue;
    }

    public static double measureGini(int[][] arr){

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


        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }


        //Iterates through the class count array, getting the gini index value for each class value
        //by taking the class count divided by the number of instances, all to be squared after and added
        //to the total gini index value.
        double localGiniIndex = 0;
        for (int i = 0; i < classCount.length; i++) {
            localGiniIndex += (Math.pow(classCount[i] / numInstances, 2));

        }
        double rootGiniIndex = 1 - localGiniIndex;

        //Now the attribute values are iterated over, getting the gini values for each of the attribute values
        //in the same manner as was done for the class values
        for (int i = 0; i < attValues.size(); i++) {

            int[] attClassCount = new int[classCount.length];

            //recalculates the class count array but for each attribute value now instead of the whole dataset
            for (int[] ints: arr) {
                if(ints[0] == attValues.get(i)){
                    attClassCount[classValues.indexOf(ints[1])]++;
                }
            }

            //gets the gini value for each count value in the attribute class count array
            for(int k : attClassCount){
                attGinis[i] += Math.pow((double)k / attCount[i], 2);
            }

            attGinis[i] = 1 - attGinis[i];

            if(Double.isNaN(attGinis[i])){
                attGinis[i] = 0;
            }
        }

        //the root gini index has each of the attribute gini values subtracted from it to give the final gini index value
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


        for (int[] ints: arr) {
            attCount[attValues.indexOf(ints[0])]++;
            classCount[classValues.indexOf(ints[1])]++;
            numInstances++;
        }

        //creates a 2D array of class predicitions, to store the class predictions for each attribute value in the data
        double[][] classPreds = new double[classCount.length][attCount.length];

        for (int i = 0; i < classCount.length; i++) {
            //the class probability is calculated by taking the count for each class value and dividing it by the number
            //of instances
            double classProb = classCount[i] / numInstances;

            //Then, for each attribute value, the predicted values for the class of each attribute value is calculated
            //and stored appropriately in the 2D array, this generates the confusion matrix used in the Chi-Squared
            //calculations
            for (int j = 0; j < attCount.length; j++) {
                classPreds[i][j] = attCount[j] * classProb;
            }
        }

        double chiSquaredIndex = 0;

        //Now the attribute values are iterated over, getting the chi-squared values for each of the attribute values
        //in the same manner as was done for the class values
        for (int i = 0; i < attCount.length; i++) {

            int[] attClassCount = new int[classCount.length];

            //recalculates the class count array but for each attribute value now instead of the whole dataset
            for (int[] ints: arr) {
                if(ints[0] == attValues.get(i)){
                    attClassCount[classValues.indexOf(ints[1])]++;
                }
            }

            //For each of the attribute values with its class count array, the class ratio is calculated
            //by taking the real class counts for each value and subtracting the predicted class counts, this figure is
            //then squared and divided by the prediction count again to give a squared ratio for each class value for the
            //attribute.
            for (int j = 0; j < attClassCount.length; j++) {
                double classRatio = Math.pow(attClassCount[j] - classPreds[j][i], 2) / classPreds[j][i];

                //The class ration is then added cumulatively to the chiSquaredIndex value, which is then returned.
                chiSquaredIndex += classRatio;
            }
        }

        return chiSquaredIndex;
    }


    public static void main(String[] args) throws Exception {

        String testDataLocation = "src\\main\\java\\ml_6002b_coursework\\test_data\\Whisky_TRAIN.arff";

        Instances test = DatasetLoading.loadData(testDataLocation);

        IGAttributeSplitMeasure igAtt = new IGAttributeSplitMeasure();

        IGAttributeSplitMeasure igAttRatio = new IGAttributeSplitMeasure();
        igAttRatio.setUseGain(false);

        ChiSquaredAttributeSplitMeasure chiAtt = new ChiSquaredAttributeSplitMeasure();

        GiniAttributeSplitMeasure giniAtt = new GiniAttributeSplitMeasure();

        System.out.println("Measure Information Gain for attribute " + test.attribute(0).name() + " splitting diagnosis = " + igAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println("Measure Information Gain Ratio for attribute " + test.attribute(0).name() + " splitting diagnosis = " + igAttRatio.computeAttributeQuality(test, test.attribute(0)));

        System.out.println("Measure Chi-Squared for attribute " + test.attribute(0).name() + " splitting diagnosis = " + chiAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println("Measure Gini Index for attribute " + test.attribute(0).name() + " splitting diagnosis = " + giniAtt.computeAttributeQuality(test, test.attribute(0)));

    }
}
