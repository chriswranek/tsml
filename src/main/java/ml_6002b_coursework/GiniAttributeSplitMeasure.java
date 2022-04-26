package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instances;

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure{
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

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
