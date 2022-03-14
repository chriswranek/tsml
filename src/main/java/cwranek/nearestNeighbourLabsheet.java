package cwranek;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.matrix.LinearRegression;

import java.io.IOException;
import java.util.Arrays;

public class nearestNeighbourLabsheet {






    public static void main(String[] args) throws Exception {

        Instances data = DatasetLoading.loadData("C:\\Users\\block\\Desktop\\Machine Learning\\FootballPlayers.arff");
        Instances[] split = InstanceTools.resampleInstances(data, 0, 0.7);


        System.out.println(data.numAttributes());
        System.out.println(data.numInstances());
        System.out.println(data.numClasses());

        System.out.println(" ");
        System.out.println(" ");

        OneNN oneNN = new OneNN();

        oneNN.buildClassifier(split[0]);

        oneNN.setkNUmber(100);

        oneNN.classifyInstance(split[1].firstInstance());
        System.out.println(100 + " = " + ClassifierTools.accuracy(split[1], oneNN));

        System.out.println(Arrays.toString(oneNN.distributionForInstance(split[1].firstInstance())));





        System.out.println(" ");
        System.out.println(" ");

        IB1 ib1 = new IB1();

        ib1.buildClassifier(split[0]);

        System.out.println(ClassifierTools.accuracy(split[1], ib1));

        System.out.println(" ");
        System.out.println(" ");

        IBk ibk = new IBk();

        ibk.buildClassifier(split[0]);

        System.out.println(ClassifierTools.accuracy(split[1], ibk));

        System.out.println(" ");
        System.out.println(" ");

        SMO smo = new SMO();

        smo.buildClassifier(split[0]);

        System.out.println(ClassifierTools.accuracy(split[1], smo));

        System.out.println(" ");
        System.out.println(" ");

        MultilayerPerceptron mlu = new MultilayerPerceptron();

        mlu.buildClassifier(split[0]);

        System.out.println(ClassifierTools.accuracy(split[1], mlu));




    }
}
