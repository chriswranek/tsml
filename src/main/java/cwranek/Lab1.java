package cwranek;

import experiments.data.DatasetLoading;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropySplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;


public class Lab1 {

    public static Instances loadData(String filePath) throws IOException {

        Instances train = null;

        try {
            FileReader reader = new FileReader(filePath);
            train = new Instances(reader);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return train;

    }

    public static void main(String[] args) throws Exception {

        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Arsenal_TEST.arff";
        String trainDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Arsenal_TRAIN.arff";
        String heightDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Height_TRAIN.arff";
        String breastDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\breast-cancer.arff";
        String golfDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\Playgolf_TRAIN.arff";

        Instances train = loadData(trainDataLocation);
        Instances test = loadData(testDataLocation);
        Instances height = loadData(heightDataLocation);

        System.out.println(test.numInstances());

        System.out.println(" ");

        System.out.println(test.numAttributes());

        System.out.println(" ");

        int winCount = 0;

        for (Instance i : train) {
            if(i.value(3) == 2){
                winCount++;
            }

            //System.out.println(i.attribute(3));
            //System.out.println(i.value(3));
        }

        System.out.println(winCount);

        System.out.println(" ");

        System.out.println(Arrays.toString(test.instance(4).toDoubleArray()));

        System.out.println(" ");

        for (Instance i : train) {
            System.out.println(i.toString());
        }

        System.out.println(" ");

        train.deleteAttributeAt(2);
        test.deleteAttributeAt(2);

        for (Instance i : train) {
            System.out.println(i.toString());
        }

        System.out.println(" ");

        train = loadData(trainDataLocation);
        test = loadData(testDataLocation);

        //for (Instance i : train) {
        //    System.out.println(i.toString());
        //}

        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(test.numAttributes()-1);
        height.setClassIndex(height.numAttributes()-1);


        NaiveBayes classifierBayes = new NaiveBayes();
        int predCount = 0;
        classifierBayes.buildClassifier(train);

        for (Instance i : train) {
            double pred = classifierBayes.classifyInstance(i);
            double actual = i.classValue();

            System.out.println("Actual = " + actual + " Predicted = " + pred);
            if(pred == actual){
                predCount++;
            }
        }

        System.out.println("Number correct = " + predCount);
        System.out.println("Accuracy = " + predCount / (double)train.numInstances());

        System.out.println(" ");

        for (Instance i : train) {
            System.out.println(Arrays.toString(classifierBayes.distributionForInstance(i)));
        }

        System.out.println(" ");

        IBk ibkClassifier = new IBk();
        int accCount = 0;
        ibkClassifier.buildClassifier(train);

        for (Instance i : train) {
            double pred = ibkClassifier.classifyInstance(i);
            double actual = i.classValue();

            System.out.println("Actual = " + actual + " Predicted = " + pred);
            if(pred == actual){
                accCount++;
            }
        }

        System.out.println("Number correct = " + accCount);
        System.out.println("Accuracy = " + accCount / (double)train.numInstances());

        System.out.println(" ");

        for (Instance i : train) {
            System.out.println(Arrays.toString(ibkClassifier.distributionForInstance(i)));
        }

        System.out.println(" ");
        System.out.println(" ");
        System.out.println(" ");

        HistogramClassifier histoClass = new HistogramClassifier();
        int histoAccCount = 0;

        histoClass.buildClassifier(height);

        for (Instance i : height) {
            double pred = histoClass.classifyInstance(i);
            double actual = i.classValue();

            System.out.println("Actual = " + actual + " Predicted = " + pred);
            if(pred == actual){
                histoAccCount++;
            }
        }

        System.out.println("Number correct = " + histoAccCount);
        System.out.println("Accuracy = " + histoAccCount / (double)height.numInstances());

        System.out.println(" ");
        System.out.println(" ");

        for (Instance i : height) {
            System.out.println(Arrays.toString(histoClass.distributionForInstance(i)));
        }

        System.out.println(" ");
        System.out.println(" ");

        WekaTools tools = new WekaTools();

        System.out.println(Arrays.toString(tools.classDistribution(train)));

        System.out.println(" ");
        System.out.println(" ");

        String basePath = "src/main/java/experiments/data/tsc/";
        String dataset = "ItalyPowerDemand";
        int seed = 1;

        Instances ItalyPowerDemand;
        RandomForest randomForest = new RandomForest();

        ItalyPowerDemand = DatasetLoading.loadDataThrowable(basePath + dataset + "/" + dataset + "_TRAIN.arff");
        System.out.println("train.relationName() = " + ItalyPowerDemand.relationName());
        System.out.println("train.numInstances() = " + ItalyPowerDemand.numInstances());
        System.out.println("train.numAttributes() = " + ItalyPowerDemand.numAttributes());
        System.out.println("train.numClasses() = " + ItalyPowerDemand.numClasses());

        randomForest.buildClassifier(ItalyPowerDemand);

        int[] predicted = tools.classifyInstances(randomForest, ItalyPowerDemand);
        int[] actual = tools.getClassValues(ItalyPowerDemand);

        System.out.println(Arrays.deepToString(tools.confusionMatrix(predicted, actual)));



        Instances newTrain = tools.loadClassificationData(trainDataLocation);

        Instances[] newTrainSplit = tools.splitData(newTrain, 0.7);

        MajorityClassClassifier majorityClassClassifier = new MajorityClassClassifier();

        majorityClassClassifier.buildClassifier(newTrain);

        int[] newPred = tools.classifyInstances(majorityClassClassifier, newTrain);
        int[] newActual = tools.getClassValues(newTrain);

        System.out.println(Arrays.deepToString(tools.confusionMatrix(newPred, newActual)));


        System.out.println(tools.accuracy(majorityClassClassifier, newTrain));

        System.out.println(" ");
        System.out.println(" ");

        ZeroR zeroR = new ZeroR();

        zeroR.buildClassifier(newTrain);

        int[] zeroRPred = tools.classifyInstances(zeroR, newTrain);
        int[] zeroRActual = tools.getClassValues(newTrain);

        System.out.println(Arrays.deepToString(tools.confusionMatrix(zeroRPred, zeroRActual)));

        System.out.println(tools.accuracy(zeroR, newTrain));


        //***********************************************************************************************************\\
        //*****************************************Labsheet 3 decision trees*****************************************\\


        Instances breastTrain = tools.loadClassificationData(breastDataLocation);

        J48 j48 = new J48();

        j48.buildClassifier(breastTrain);

        int[] j48pred = tools.classifyInstances(j48, breastTrain);

        System.out.println(Arrays.toString(j48pred));

        System.out.println(j48.measureTreeSize());

        j48.setBinarySplits(true);

        j48pred = tools.classifyInstances(j48, breastTrain);

        System.out.println(Arrays.toString(j48pred));

        j48.setReducedErrorPruning(true);

        j48pred = tools.classifyInstances(j48, breastTrain);

        System.out.println(Arrays.toString(j48pred));

        System.out.println(" ");
        System.out.println(" ");

        System.out.println(j48);


        System.out.println(" ");
        System.out.println(" ");

        Instances golfTrain = tools.loadClassificationData(golfDataLocation);

        //Distribution distro = new Distribution(golfTrain);

        //InfoGainSplitCrit infoGain = new InfoGainSplitCrit();

        //System.out.println(infoGain.splitCritValue(distro));



        J48 c45 = new J48();

        c45.buildClassifier(golfTrain);

        System.out.println(c45);

        System.out.println(" ");
        System.out.println(" ");


        double[][] outlook = new double[golfTrain.attribute("Outlook").numValues()][golfTrain.numClasses()];

        for (Instance ins : golfTrain){
            outlook[(int)ins.value(0)][(int)ins.classValue()]++;
        }

        for(double[] x : outlook){
            for(double y : x){
                System.out.print(y + ",");
            }
            System.out.print("\n");
        }

        //int[][] temp = new int[golfTrain.attribute("Temp").numValues()][golfTrain.numClasses()];

        //for (Instance ins : golfTrain){
        //     temp[(int)ins.value(0)][(int)ins.classValue()]++;
        //}

        InfoGainSplitCrit infoGain = new InfoGainSplitCrit();

        Distribution distro = new Distribution(outlook);

        System.out.println("Distro for outlook = " + distro.dumpDistribution());

        double outlookInfoGain = infoGain.splitCritValue(distro);

        System.out.println("Outlook info gain = " + 1 / outlookInfoGain);

        System.out.println(" ");
        System.out.println(" ");

        EntropySplitCrit entroGain = new EntropySplitCrit();

        Distribution newDistro = new Distribution(outlook);

        System.out.println("Distro for outlook = " + newDistro.dumpDistribution());

        outlookInfoGain = 0;

        outlookInfoGain = entroGain.splitCritValue(newDistro);

        System.out.println("Outlook entropy gain = " + 1 / outlookInfoGain);

        System.out.println(" ");
        System.out.println(" ");

        Bagging bag = new Bagging();

        bag.setClassifier(c45);

    }

}
