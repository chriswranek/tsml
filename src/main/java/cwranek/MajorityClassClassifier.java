package cwranek;

import java.util.Arrays;
import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class MajorityClassClassifier extends AbstractClassifier {

    private double classValue;
    private double[] classCount;
    private Attribute classAtt;

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        //instances = new Instances(instances);
        //instances.deleteWithMissingClass();

        double sumOfClassWeights = 0;

        classAtt = instances.classAttribute();
        classValue = 0;

        switch (instances.classAttribute().type()){
            case Attribute.NUMERIC:
                classCount = null;
                break;
            case Attribute.NOMINAL:
                classCount = new double[instances.numClasses()];
                Arrays.fill(classCount, 1);
                sumOfClassWeights = instances.numClasses();
                break;
        }


        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            if(instances.classAttribute().isNominal()){
                classCount[(int) instance.classValue()] += instance.weight();
            } else {
                classValue += instance.weight() * instance.classValue();
            }
            sumOfClassWeights += instance.weight();
        }

        if (instances.classAttribute().isNumeric()){
            if(Utils.gr(sumOfClassWeights, 0)){
                classValue /= sumOfClassWeights;
            }
        } else {
            classValue = Utils.maxIndex(classCount);
            Utils.normalize(classCount, sumOfClassWeights);
        }

    }

    public double classifyInstance(Instance instance) {
        return classValue;
    }

    public double [] distributionForInstance(Instance instance)
            throws Exception {

        if(classCount == null){
            double[] result = new double[1];
            result[0] = classValue;
            return result;
        } else {
            return (double []) classCount.clone();
        }
    }
}
