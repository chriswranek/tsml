package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    int numTrees = 50;
    double splitProportion = 0.5;
    boolean averageDistributions = false;
    ID3Coursework[] treeEnsemble;
    double[] classDistro;
    double attProp = 0.5;

    @Override
    public void buildClassifier(Instances data) throws Exception {

        treeEnsemble = new ID3Coursework[numTrees];

        int max = 3;
        for (int i = 0; i < numTrees; i++) {
            treeEnsemble[i] = new ID3Coursework();
            Random r = new Random();

            treeEnsemble[i].setOptions(r.nextInt(max));
        }

        for (int i = 0; i < numTrees; i++) {

            Instances tempData = new Instances(data);

            for (int j = 0; j < (int)(data.numAttributes() * (1 - attProp)); j++) {
                Random rand = new Random();

                Attribute index = tempData.attribute(rand.nextInt(data.numAttributes()));

                tempData.deleteWithMissing(index);
            }

            treeEnsemble[i].buildClassifier(tempData);
        }

        classDistro = new double[data.numClasses()];
    }

    public void buildRandomSubspace(Instances data, double attProp) throws Exception {

        treeEnsemble = new ID3Coursework[numTrees];

        for (int i = 0; i < numTrees; i++) {
            treeEnsemble[i] = new ID3Coursework();
        }

        for (int i = 0; i < numTrees; i++) {

            Instances tempData = new Instances(data);

            for (int j = 0; j < (int)(data.numAttributes() * attProp); j++) {
                Random rand = new Random();

                Attribute index = tempData.attribute(rand.nextInt(data.numAttributes()));

                tempData.deleteWithMissing(index);
            }

            treeEnsemble[i].buildClassifier(tempData);
        }

        classDistro = new double[data.numClasses()];
    }

    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

        double[] classPreds = new double[classDistro.length];

        for (int i = 0; i < numTrees; i++) {
            classPreds[(int) treeEnsemble[i].classifyInstance(instance)]++;
        }

        return findLargestVal(classPreds);
    }


    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {

        classDistro = new double[classDistro.length];

        if(averageDistributions){
            for (int i = 0; i < numTrees; i++) {
                double[] tempDistro = treeEnsemble[i].distributionForInstance(instance);

                for (int j = 0; j < tempDistro.length; j++) {
                    classDistro[j] += tempDistro[j];
                }
            }
        } else {
            for (int i = 0; i < numTrees; i++) {
                classDistro[(int) treeEnsemble[i].classifyInstance(instance)]++;
            }
        }

        for (int i = 0; i < classDistro.length; i++) {
            classDistro[i] = classDistro[i] / numTrees;
        }

        return classDistro;
    }


    public int findLargestVal(double[] arr){
        int largestIndex = 0;

        for (int j = 1; j < arr.length; j++) {
            if(arr[j] > arr[largestIndex]){
                largestIndex = j;
            }
        }

        return largestIndex;
    }

    public void setNumTrees(int numTrees){
        this.numTrees = numTrees;
    }


    public static void main(String[] args) throws Exception {

        String optDigitsDataset = "src\\main\\java\\ml_6002b_coursework\\test_data\\optdigits.arff";

        Instances optDigitsInstances = DatasetLoading.loadData(optDigitsDataset);

        Instances[] trainTestSplit = InstanceTools.resampleInstances(optDigitsInstances, 0, Math.random());

        TreeEnsemble optTreeEnsemble = new TreeEnsemble();

        optTreeEnsemble.buildClassifier(trainTestSplit[0]);

        System.out.println("TreeEnsemble on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optTreeEnsemble));

        for (int i = 0; i < 5; i++) {
            System.out.println(Arrays.toString(optTreeEnsemble.distributionForInstance(trainTestSplit[1].get(i))));
        }

        System.out.println(" ");
        System.out.println(" ");

        String chinaTownDatasetTrain = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TRAIN.arff";
        String chinaTownDatasetTest = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TEST.arff";

        Instances chinaTownTrain = DatasetLoading.loadData(chinaTownDatasetTrain);
        Instances chinaTownTest = DatasetLoading.loadData(chinaTownDatasetTest);

        Instances discretizedChinaTownTrain = Discretize.discretizeDataset(chinaTownTrain);
        Instances discretizedChinaTownTest  = Discretize.discretizeDataset(chinaTownTest);

        TreeEnsemble chinaEnsemble = new TreeEnsemble();

        chinaEnsemble.buildClassifier(discretizedChinaTownTrain);

        System.out.println("TreeEnsemble on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(discretizedChinaTownTest, chinaEnsemble));

        for (int i = 0; i < 5; i++) {
            System.out.println(Arrays.toString(chinaEnsemble.distributionForInstance(discretizedChinaTownTest.get(i))));
        }
    }
}
