package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;

public class IGAttributeSplitMeasure implements AttributeSplitMeasure{

    private boolean useGain = true;

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

        int[][] computeArr = new int[data.numInstances()][2];

        for (int i = 0; i < data.numInstances(); i++) {
            computeArr[i][0] = (int) data.get(i).value(att);
            computeArr[i][1] = (int) data.get(i).classValue();
        }

        //System.out.println(Arrays.deepToString(computeArr));

        if(useGain){
            return AttributeMeasures.measureInformationGain(computeArr);
        } else {
            return AttributeMeasures.measureInformationGainRatio(computeArr);
        }
    }

    public void setUseGain(boolean newGain){
        useGain = newGain;
    }




    public static void main(String[] args) throws Exception {
        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Whisky_TRAIN.arff";

        Instances test = DatasetLoading.loadData(testDataLocation);

        IGAttributeSplitMeasure IGAtt = new IGAttributeSplitMeasure();


        System.out.println("Measure Information Gain for attribute " + test.attribute(0).name() + " splitting diagnosis = " + IGAtt.computeAttributeQuality(test, test.attribute(0)));

        System.out.println("Measure Information Gain for attribute " + test.attribute(1).name() + " splitting diagnosis = " + IGAtt.computeAttributeQuality(test, test.attribute(1)));

        System.out.println("Measure Information Gain for attribute " + test.attribute(2).name() + " splitting diagnosis = " + IGAtt.computeAttributeQuality(test, test.attribute(2)));

    }

}
