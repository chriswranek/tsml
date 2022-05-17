package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 *
 */
public interface AttributeSplitMeasure {

    double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
     default Instances[] splitData(Instances data, Attribute att) {

         Instances[] splitData = new Instances[att.numValues()];
         for (int j = 0; j < att.numValues(); j++) {
             splitData[j] = new Instances(data, data.numInstances());
         }
         Enumeration instEnum = data.enumerateInstances();
         while (instEnum.hasMoreElements()) {
             Instance inst = (Instance) instEnum.nextElement();
             splitData[(int) inst.value(att)].add(inst);
         }
         for (Instances splitDatum : splitData) {
             splitDatum.compactify();
         }
         return splitData;
    }





    /**
     * Splits a dataset according to the values of a numeric attribute.
     *
     * @param data the data which is to be split
     * @param value the numeric value to be used for binary splitting
     * @param discretized boolean for whether the data has been discretized before splitting
     * @return the sets of instances produced by the split above and below the value
     */
    default Instances[] splitDataOnNumeric(Instances data, Attribute att, double value, boolean discretized, int numOfBins) {


        //If the data has been discretized before splitting then the data is split according to the number of bins the
        //data has been discretized into
        if(discretized){
            Instances[] splitData = new Instances[numOfBins];
            for (int j = 0; j < numOfBins; j++) {
                splitData[j] = new Instances(data, data.numInstances());
            }
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                splitData[(int) inst.value(att)].add(inst);
            }
            for (Instances splitDatum : splitData) {
                splitDatum.compactify();
            }
            return splitData;
        } else {
            double meanValue = 0;

            //If the value given to the function is 0, then a mean value will be calculated for all the attribute values
            //to determine the best value to perform the binary split on
            if(value == 0){
                for (int i = 0; i < data.numInstances(); i++) {
                    meanValue += data.get(i).value(att);
                }
                meanValue /= data.numInstances();
                value = meanValue;
            }

            //Binary Split
            Instances[] splitData = new Instances[2];

            for (int i = 0; i < splitData.length; i++) {
                splitData[i] = new Instances(data, data.numInstances());
            }

            //The data is split into two seperate instances, and each attribute value is assessed to see whether it falls
            //above or below the chosen splitting value
            for (int i = 0; i < data.numInstances(); i++) {
                if(data.get(i).value(att) < value){
                    splitData[0].add(data.get(i));
                } else {
                    splitData[1].add(data.get(i));
                }
            }

            for (Instances splitDatum : splitData) {
                splitDatum.compactify();
            }
            return splitData;
        }
    }

}
