package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class TestHarness {

    public static int[] getAttributeIndices(ArrayList<Integer> arrayList, int numOfAttributes, double attProportion){

        int[] attIndicesArr = new int[(int) (numOfAttributes * (1 - attProportion))];
        Random rand = new Random();

        while(arrayList.size() != attIndicesArr.length){
            int attIndex = rand.nextInt(numOfAttributes-1);
            if(attIndex != 0) {
                if (!arrayList.contains(attIndex)) {
                    arrayList.add(attIndex);
                    numOfAttributes--;
                }
            }
        }

        for (int i = 0; i < arrayList.size(); i++) {
            attIndicesArr[i] = arrayList.get(i);
        }

        return attIndicesArr;
    }

    public static void main(String[] args) throws Exception {


        String[] str = {"C:\\Experiments\\Results\\Case Study\\","C:\\Users\\block\\Documents\\GitHub\\tsml\\src\\" +
                "main\\java\\ml_6002b_coursework\\test_data\\Case Study\\","30","false","TreeEnsemble","0"};

        experiments.CollateResults.collate(str);


        //System.out.println(result);

        /*
        String chinaTownDatasetTrain = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TRAIN.arff";
        String chinaTownDatasetTest = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TEST.arff";

        Instances chinaTownTrain = DatasetLoading.loadData(chinaTownDatasetTrain);
        Instances chinaTownTest = DatasetLoading.loadData(chinaTownDatasetTest);

        System.out.println(chinaTownTrain);

        Instances discretizedChinaTownTrain = Discretize.discretizeDataset(chinaTownTrain);
        Instances discretizedChinaTownTest  = Discretize.discretizeDataset(chinaTownTest);

        Instances newChinaTownTrain;
        Instances newChinaTownTest;

        System.out.println(discretizedChinaTownTest);

        NumericToNominal numericTrain = new NumericToNominal();
        NumericToNominal numericTest = new NumericToNominal();

        int[] attIndices = new int[discretizedChinaTownTrain.numAttributes()];

        for (int i = 0; i < discretizedChinaTownTrain.numAttributes(); i++) {
            attIndices[i] = i;
        }

        numericTrain.setAttributeIndicesArray(attIndices);

        numericTrain.setInputFormat(discretizedChinaTownTrain);
        numericTest.setInputFormat(discretizedChinaTownTest);



        newChinaTownTrain = Filter.useFilter(discretizedChinaTownTrain, numericTrain);
        newChinaTownTest = Filter.useFilter(discretizedChinaTownTest, numericTest);


        System.out.println(newChinaTownTest);

        TreeEnsemble chinaEnsemble = new TreeEnsemble();

        ID3Coursework tree = new ID3Coursework();

        //tree.buildClassifier(newChinaTownTrain);

        chinaEnsemble.buildClassifier(newChinaTownTrain);

        System.out.println("TreeEnsemble on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(newChinaTownTest, chinaEnsemble));
        */

        /*
        String optDigitsDataset = "src\\main\\java\\ml_6002b_coursework\\test_data\\optdigits.arff";

        Instances optDigitsInstances = DatasetLoading.loadData(optDigitsDataset);

        System.out.println(optDigitsInstances);

        ArrayList<Integer> arrayList = new ArrayList<>();

        int[] attArr = getAttributeIndices(arrayList, optDigitsInstances.numAttributes() - 1, 0.5);

        System.out.println(Arrays.toString(attArr));

        for (int attIndex: attArr) {
            optDigitsInstances.deleteAttributeAt(attIndex);
        }

        System.out.println(optDigitsInstances);

        Instances[] trainTestSplit = InstanceTools.resampleInstances(optDigitsInstances, 0, 0.7);

        TreeEnsemble optTreeEnsemble = new TreeEnsemble();

        optTreeEnsemble.buildClassifier(trainTestSplit[0]);

        System.out.println("TreeEnsemble on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optTreeEnsemble));
        */

        Bagging bagging = new Bagging();

        BayesNet bayesNet = new BayesNet();

        DecisionStump decisionStump = new DecisionStump();

        DecisionTable decisionTable = new DecisionTable();

        HoeffdingTree hoeffdingTree = new HoeffdingTree();

        NaiveBayes naiveBayes = new NaiveBayes();

        RandomForest randomForest = new RandomForest();

        RotationForest rotationForest = new RotationForest();

        SimpleCart simpleCart = new SimpleCart();

        ZeroR zeroR = new ZeroR();


    }
}
