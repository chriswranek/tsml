package cwranek;

import core.contracts.Dataset;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import fileIO.OutFile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.*;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class ensembles_lab4 {

static String[] problems={
    "bank",
    "blood",
    "breast-cancer-wisc-diag",
    "breast-tissue",
    "cardiotocography-10clases",
    "conn-bench-sonar-mines-rocks",
    "conn-bench-vowel-deterding",
     "ecoli",
    "glass",
    "hill-valley",
    "image-segmentation",
    "ionosphere",
    "iris",
    "libras",
    "optical",
    "ozone",
    "page-blocks",
        "parkinsons",
    "planning",
    "post-operative",
    "ringnorm",
    "seeds",
    "spambase",
    "statlog-landsat",
    "statlog-vehicle",
    "steel-plates",
    "synthetic-control",
    "twonorm",
    "vertebral-column-3clases",
    "wall-following",
    "waveform-noise",
    "wine-quality-white",
     "yeast"};



    public static void baggingExperiments() throws Exception {
        Instances train,test;
        String path= "C:\\Temp\\UCIContinuous\\";
        System.out.println(" number of problems = "+problems.length);
        double meanDiff=0;
        double meanB=0, meanRF=0;
        int count=0;
        for(String str:problems){
            train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            Bagging b= new Bagging();
            b.setNumIterations(500);
            RandomForest rf = new RandomForest();
            rf.setNumTrees(500);
            b.buildClassifier(train);
            rf.buildClassifier(train);
            double accB=ClassifierTools.accuracy(test,b);
            double accRF=ClassifierTools.accuracy(test,rf);
            meanDiff+=accRF-accB;
            meanB+=accB;
            meanRF+=accRF;
            if(accRF>accB)
                count++;
            System.out.println(str+ " Bagging = "+accB+" RandF  = "+accRF);
        }
        meanDiff/=problems.length;
        meanB/=problems.length;
        meanRF/=problems.length;
        System.out.println("RF wins "+count++);
        System.out.println("Mean Diff "+meanDiff);
        System.out.println("Mean B "+meanB);
        System.out.println("Mean RF  "+meanRF);

    }

    public static void j48experiment() throws Exception {
        Instances train, test;
        String path= "C:\\Temp\\UCIContinuous\\";
        System.out.println(" number of problems = "+problems.length);
        double meanRF = 0;
        double meanJ48 = 0;
        for(String str:problems){
            train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
            test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
            RandomForest rf = new RandomForest();
            J48 j48 = new J48();
            rf.setNumTrees(500);
            j48.buildClassifier(train);
            rf.buildClassifier(train);
            double accJ48=ClassifierTools.accuracy(test, j48);
            double accRF=ClassifierTools.accuracy(test,rf);
            //System.out.println(str+ " RandF  = "+accRF);
            meanRF+=accRF;
            meanJ48+=accJ48;
        }
        meanRF /= problems.length;
        meanJ48 /= problems.length;
        System.out.println("Mean accuracy of random forest for 500 trees = " + meanRF );
        System.out.println("Mean accuracy of J48 = " + meanJ48 );
    }

    public static void randomForestExperiment() throws Exception {
        Instances train, test;
        String path= "C:\\Temp\\UCIContinuous\\";
        System.out.println(" number of problems = "+problems.length);

        for (int i = 10; i <= 90; i+=10) {
            double meanRF = 0;
            for(String str:problems){
                train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
                test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
                RandomForest rf = new RandomForest();
                rf.setNumTrees(i);
                rf.buildClassifier(train);
                double accRF=ClassifierTools.accuracy(test,rf);
                //System.out.println(str+ " RandF  = "+accRF);
                meanRF+=accRF;
            }
            meanRF /= problems.length;
            System.out.println("Mean accuracy of random forest for " + i + " trees = " + meanRF );
        }

        for (int i = 100; i <= 1000; i+=100) {
            double meanRF = 0;
            for(String str:problems){
                train = DatasetLoading.loadData(path+str+"\\"+str+"_TRAIN.arff");
                test = DatasetLoading.loadData(path+str+"\\"+str+"_TEST.arff");
                RandomForest rf = new RandomForest();
                rf.setNumTrees(i);
                rf.buildClassifier(train);
                double accRF=ClassifierTools.accuracy(test,rf);
                //System.out.println(str+ " RandF  = "+accRF);
                meanRF+=accRF;
            }
            meanRF /= problems.length;
            System.out.println("Mean accuracy of random forest for " + i + " trees = " + meanRF );
        }
    }

    public static void logitBoostExample() throws Exception {
        String pth ="C:\\Temp\\UCIContinuous\\";
        Instances train = DatasetLoading.loadData(pth+"bank\\bank.arff");
        System.out.println(train);
        LogitBoost logit = new LogitBoost();
        logit.setDebug(true);

        logit.buildClassifier(train);
    }

    public static void boostingExperiments() throws Exception {
        String pth ="C:\\Temp\\UCIContinuous\\";
        Instances train = DatasetLoading.loadData(pth+"adiac\\Adiac_TRAIN.arff");
        Instances test = DatasetLoading.loadData(pth+"adiac\\Adiac_TEST.arff");
        AdaBoostM1 ada= new AdaBoostM1();
        System.out.println(" Resampling ="+ada.getUseResampling());
        ada.setUseResampling(true);
        ada.buildClassifier(train);
        double adaAcc = ClassifierTools.accuracy(test, ada);

        System.out.println(adaAcc);

        System.out.println(" ");
        System.out.println(" ");

        LogitBoost logit = new LogitBoost();
        logit.setDebug(true);
        logit.buildClassifier(train);
        double logitAcc = ClassifierTools.accuracy(test, logit);

        System.out.println(logitAcc);

        System.out.println(" ");
        System.out.println(" ");

        J48 j48 = new J48();
        j48.buildClassifier(train);
        double j48Acc = ClassifierTools.accuracy(test, j48);

        System.out.println(j48Acc);
    }

    public static void ensembleExperiment() throws Exception {
        WekaTools tools = new WekaTools();
        EnsembleClassifier ensemble = new EnsembleClassifier();
        String pth ="C:\\Temp\\UCIContinuous\\";
        Instances train = DatasetLoading.loadData(pth+"adiac\\Adiac_TRAIN.arff");
        Instances test = DatasetLoading.loadData(pth+"adiac\\Adiac_TEST.arff");
        double acc = 0;

        ensemble.buildClassifier(train);

        acc = ClassifierTools.accuracy(test, ensemble);

        //acc/= ensemble.getM_numOfClassifiers();
        System.out.println(acc);

    }

    public static void redVBlackLabelExperiment() throws Exception {

        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\JW_RedVsBlack0_TEST.arff";
        String trainDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\JW_RedVsBlack0_TRAIN.arff";

        Instances train = DatasetLoading.loadData(trainDataLocation);
        Instances test = DatasetLoading.loadData(testDataLocation);

        Classifier c = new NaiveBayes();

        c.buildClassifier(train);

        OutFile out = new OutFile("C:\\Users\\block\\Desktop\\Machine Learning\\naiveOutput.csv");

        out.writeLine(c.getClass().getSimpleName()+","+"Red vs Black Label");
        out.writeLine("No parameter info");
        out.writeLine(String.valueOf(ClassifierTools.accuracy(test,c)));

        for (Instance i : test){

            int pred = (int)c.classifyInstance(i);
            double [] probs = c.distributionForInstance(i);
            out.writeString((int)i.classValue()+","+pred+",");
            System.out.print((int)i.classValue()+","+pred+",");
            for(double d : probs){
                System.out.print(","+d);
                out.writeString(","+d);
            }
            System.out.println("\n");
            out.writeString("\n");

        }

    }

    public static void classifyResults() throws Exception {
        ClassifierResults res = new ClassifierResults();


        res.loadResultsFromFile("C:\\Users\\block\\Desktop\\Machine Learning\\Results\\UCR\\Python\\NaiveBayes\\Predictions\\bank\\testFold0.csv");

        res.findAllStats();

        System.out.println(res.balancedAcc);
        System.out.println(Arrays.deepToString(res.confusionMatrix));
    }

    public static void testBayesians() throws Exception {
        String testDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\JW_RedVsBlack0_TEST.arff";
        String trainDataLocation = "C:\\Users\\block\\Desktop\\Machine Learning\\JW_RedVsBlack0_TRAIN.arff";

        Instances train = DatasetLoading.loadData(trainDataLocation);
        Instances test = DatasetLoading.loadData(testDataLocation);

        BayesianLogisticRegression logic = new BayesianLogisticRegression();

        BayesNet bayesNet = new BayesNet();

        DMNBtext dmnBtext = new DMNBtext();

        NaiveBayesSimple bayesSimple = new NaiveBayesSimple();

        logic.buildClassifier(train);
        bayesNet.buildClassifier(train);
        dmnBtext.buildClassifier(train);
        bayesSimple.buildClassifier(train);

        ArrayList<Classifier> classifiers = new ArrayList<>();

        classifiers.add(logic);
        classifiers.add(bayesNet);
        classifiers.add(dmnBtext);
        classifiers.add(bayesSimple);

        for (Classifier c : classifiers){
            OutFile out = new OutFile("C:\\Users\\block\\Desktop\\Machine Learning\\ClassifierEval\\" + c.getClass().getSimpleName() +"Results.csv");
            out.writeLine(c.getClass().getSimpleName()+","+"Red vs Black Label");
            out.writeLine("No parameter info");
            out.writeLine(String.valueOf(ClassifierTools.accuracy(test,c)));

            for (Instance i : test){

                int pred = (int)c.classifyInstance(i);
                double [] probs = c.distributionForInstance(i);
                out.writeString((int)i.classValue()+","+pred+",");
                System.out.print((int)i.classValue()+","+pred+",");
                for(double d : probs){
                    System.out.print(","+d);
                    out.writeString(","+d);
                }
                System.out.println("\n");
                out.writeString("\n");

            }
        }


    }


    public static void crossEval() throws Exception {
        String problem = "Whisky";
        Instances data = DatasetLoading.loadData("C:\\Users\\block\\Desktop\\Machine Learning\\JW_RedVsBlack0_TRAIN.arff");
        Instances[] split = InstanceTools.resampleInstances(data, 0, 0.7);
        Evaluation eval = new Evaluation(split[0]);

        Classifier c = new J48();

        c.buildClassifier(split[0]);
        eval.evaluateModel(c, split[1]);
        double acc = 1 - eval.errorRate();
        double weightAuroc = eval.weightedAreaUnderROC();

        System.out.println("Acc = "+acc+" AUROC = "+weightAuroc);

        eval.crossValidateCustomModel(c, data, 10, new Random());
        acc = 1- eval.errorRate();
    }

    public static void naiveVsNet() throws Exception {

        String path= "C:\\Temp\\UCIContinuous\\";
        System.out.println(" number of problems = "+problems.length);
        NaiveBayes naiveBayes = new NaiveBayes();
        BayesNet bayesNet = new BayesNet();

        for(String str:problems){
            Instances trainTest = DatasetLoading.loadData(path+str+"\\"+str+".arff");
            Instances[] split = InstanceTools.resampleInstances(trainTest, 0, 0.7);
            naiveBayes.buildClassifier(split[0]);
            bayesNet.buildClassifier(split[0]);

            OutFile naive = new OutFile("C:\\Users\\block\\Desktop\\Machine Learning\\Results\\UCR\\Python\\NaiveBayes\\Predictions\\"+str+"\\" + "testFold0.csv");
            //naive.writeLine(naiveBayes.getClass().getSimpleName()+","+ str);
            naive.writeLine(str+","+naiveBayes.getClass().getSimpleName());
            naive.writeLine("No parameter info");

            OutFile net = new OutFile("C:\\Users\\block\\Desktop\\Machine Learning\\Results\\UCR\\Python\\BayesNet\\Predictions\\"+str+"\\" + "testFold0.csv");
            net.writeLine(str+","+bayesNet.getClass().getSimpleName());
            net.writeLine("No parameter info");





            naive.writeLine(String.valueOf(ClassifierTools.accuracy(split[1],naiveBayes)));



            net.writeLine(String.valueOf(ClassifierTools.accuracy(split[1],bayesNet)));

            for (Instance i : split[1]){

                int pred = (int)naiveBayes.classifyInstance(i);
                double [] probs = naiveBayes.distributionForInstance(i);

                int netPred = (int)bayesNet.classifyInstance(i);
                double [] netProbs = bayesNet.distributionForInstance(i);
                naive.writeString((int)i.classValue()+","+pred+",");
                net.writeString((int)i.classValue()+","+netPred+",");
                //System.out.print((int)i.classValue()+","+pred+",");
                for(double d : probs){
                    //System.out.print(","+d);
                    naive.writeString(","+d);
                }

                for(double d : netProbs){
                    //System.out.print(","+d);
                    net.writeString(","+d);
                }
                //System.out.println("\n");
                naive.writeString("\n");
                net.writeString("\n");

            }
        }
    }

    public static void main(String[] args) throws Exception {
        //logitBoostExample();
        //boostingExperiments();
        //baggingExperiments();
        //randomForestExperiment();
        //j48experiment();
        //ensembleExperiment();
        //redVBlackLabelExperiment();
        //classifyResults();
        //testBayesians();
        //crossEval();
        //naiveVsNet();
        //classifyResults();

        String[] str = {"C:\\Users\\block\\Desktop\\Machine Learning\\Results\\UCR\\Python\\","C:\\Temp\\UCIContinuous\\","1","false","NaiveBayes","0"};

        String[] netStr = {"C:\\Users\\block\\Desktop\\Machine Learning\\Results\\UCR\\Python\\","C:\\Temp\\UCIContinuous\\","1","false","bayesNet","0"};

        experiments.CollateResults.collate(str);

        experiments.CollateResults.collate(netStr);
    }
}
