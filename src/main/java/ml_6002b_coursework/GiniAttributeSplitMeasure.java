package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instances;

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure{


    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

        //Before computing the attribute quality, the data Instances need to be sorted by the chosen
        //attribute. This is done by iterating through the instances and placing the selected attribute value and its
        //corresponding class value for that instance into a 2D array. This array will have two columns and as many rows
        //as there are instances in the data. This array is then passed to attribute selection functions to compute the
        //quality of the attribute.
        int[][] computeArr = new int[data.numInstances()][2];

        for (int i = 0; i < data.numInstances(); i++) {
            computeArr[i][0] = (int) data.get(i).value(att);
            computeArr[i][1] = (int) data.get(i).classValue();
        }

        return AttributeMeasures.measureGini(computeArr);
    }




    public static void main(String[] args) throws Exception {
        String testDataLocation = "src\\main\\java\\ml_6002b_coursework\\test_data\\Whisky_TRAIN.arff";

        Instances test = DatasetLoading.loadData(testDataLocation);

        GiniAttributeSplitMeasure giniAtt = new GiniAttributeSplitMeasure();


        System.out.println("Measure Information Gain for attribute " + test.attribute(0).name() + " splitting diagnosis = " + giniAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println("Measure Information Gain for attribute " + test.attribute(1).name() + " splitting diagnosis = " + giniAtt.computeAttributeQuality(test, test.attribute(1)));

        System.out.println("Measure Information Gain for attribute " + test.attribute(2).name() + " splitting diagnosis = " + giniAtt.computeAttributeQuality(test, test.attribute(2)));

    }
}
