package ml_6002b_coursework;

import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    int numTrees = 50;
    double splitProportion = 0.5;
    boolean averageDistributions;
    ID3Coursework[] treeEnsemble;
    double[] classDistro;
    double attProp = 0.5;

    @Override
    public void buildClassifier(Instances data) throws Exception {

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

        /*
        treeEnsemble = new ID3Coursework[numTrees];

        for (int i = 0; i < numTrees; i++) {
            treeEnsemble[i] = new ID3Coursework();
        }

        //Instances[] trainTestSplit = InstanceTools.resampleInstances(data, 0, splitProportion);


        for (int i = 0; i < numTrees; i++) {
            Random rand = new Random();

            treeEnsemble[i].setOptions(rand.nextInt(3));

            treeEnsemble[i].buildClassifier(data);
        }

        classDistro = new double[data.numClasses()];
        */
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

                for (int j = 0; j < tempDistro.length; j++) { classDistro[j] += tempDistro[j]; }
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


    public static void main(String[] args) throws Exception {



        //Instances[] trainTestSplit = InstanceTools.resampleInstances(optDigitsTest, 0, 0.7);

        //TreeEnsemble treeEnsemble = new TreeEnsemble();

        //treeEnsemble.buildClassifier(trainTestSplit[0]);

        //treeEnsemble.buildRandomSubspace(trainTestSplit[0], 0.5);

        //treeEnsemble.averageDistributions = false;

        //System.out.println(ClassifierTools.accuracy(trainTestSplit[1], treeEnsemble));

        //for (int i = 0; i < 5; i++) { System.out.println(Arrays.toString(treeEnsemble.distributionForInstance(trainTestSplit[1].get(i)))); }

        //System.out.println(" ");
        //System.out.println(" ");


        //String UCIDatasetLocation = "src\\main\\java\\ml_6002b_coursework\\test_data\\UCI Discrete\\";

        //System.out.println(DatasetLists.nominalAttributeProblems.length);

        /*
        for(String str : DatasetLists.nominalAttributeProblems){

            Instances trainTest = DatasetLoading.loadData(UCIDatasetLocation+str+"\\"+str+".arff");
            Instances[] split = InstanceTools.resampleInstances(trainTest, 0, 0.7);

            double[] classCounts = new double[trainTest.numClasses()];

            for (int i = 0; i < trainTest.numInstances(); i++) {
                classCounts[(int) trainTest.get(i).classValue()]++;
            }

            System.out.println(str);
            System.out.println("Attributes : " + trainTest.numAttributes() + ", Train/Test Cases : " + split[0].numInstances() + "/" + split[1].numInstances());
            System.out.println("Num Classes : " + trainTest.numClasses() + ", Class distribution : " + Arrays.toString(classCounts));

            System.out.println(" ");
            System.out.println(" ");
        }
        */


        //for (Instance instance: trainTestSplit[1]) { System.out.println(treeEnsemble.classifyInstance(instance)); }

        /*
        String chinaTownTrain = "src\\main\\java\\ml_6002b_coursework\\test_data\\Chinatown_TRAIN.arff";
        String chinaTownTest = "src\\main\\java\\ml_6002b_coursework\\test_data\\Chinatown_TEST.arff";

        Instances chinaTownTrainData = DatasetLoading.loadData(chinaTownTrain);
        Instances chinaTownTestData = DatasetLoading.loadData(chinaTownTest);

        TreeEnsemble chinaTownTree = new TreeEnsemble();

        chinaTownTree.buildClassifier(chinaTownTrainData);

        System.out.println(ClassifierTools.accuracy(chinaTownTestData, chinaTownTree));

        for (int i = 0; i < 5; i++) {
            //treeEnsemble.distributionForInstance(trainTestSplit[1].get(i));
            System.out.println(Arrays.toString(chinaTownTree.distributionForInstance(chinaTownTestData.get(i))));
        }
        */

        //String[] str = {"C:\\Experiments\\Results\\","C:\\Users\\block\\Documents\\GitHub\\tsml\\src\\main\\java\\ml_6002b_coursework\\test_data\\UCI Discrete\\","30","false","RotF","0"};

        //experiments.CollateResults.collate(str);

        String trainDataLocation = "src\\main\\java\\ml_6002b_coursework\\test_data\\FiftyWords\\FiftyWords_TRAIN.arff";

        String testDataLocation = "src\\main\\java\\ml_6002b_coursework\\test_data\\FiftyWords\\FiftyWords_TEST.arff";

        Instances fiftyWordsTrain = DatasetLoading.loadData(trainDataLocation);
        Instances fiftyWordsTest = DatasetLoading.loadData(testDataLocation);

        for (int i = 0; i < 5; i++) {
            System.out.println(fiftyWordsTrain.get(i));
        }

        Discretize discretize = new Discretize();

        discretize.setBins(10);

        discretize.setInputFormat(fiftyWordsTrain);

        discretize.setOutputFormat();

        System.out.println(discretize.getOutputFormat());

        /*
        int instanceCount = 0;
        for (Instance inst: fiftyWordsTrain) {
            discretize.input(inst);

            fiftyWordsTrain.remove(instanceCount);

            fiftyWordsTrain.add(discretize.output());


            instanceCount++;
        }

        for (int i = 0; i < 5; i++) {
            System.out.println(fiftyWordsTrain.get(i));
        }

         */






    }
}
