package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    int numTrees = 50;
    boolean averageDistributions = false;
    ID3Coursework[] treeEnsemble;
    double[] classDistro;
    double attProp = 0.5;


    @Override
    public void buildClassifier(Instances data) throws Exception {

        treeEnsemble = new ID3Coursework[numTrees];

        //This first loop sets the attribute selection method for each of the trees in the ensemble
        //There are 4 methods to choose from, so a random number between 0 and 3 is generated and
        //setOptions is called for each tree, giving it the random number to set its selection method
        int max = 4;
        for (int i = 0; i < numTrees; i++) {
            treeEnsemble[i] = new ID3Coursework();
            Random r = new Random();

            treeEnsemble[i].setOptions(r.nextInt(max));
        }


        for (int i = 0; i < numTrees; i++) {

            //For each tree in the ensemble, a new set of instances is generated, and for each of the new instances
            //its attributes are sampled randomly, how many are sampled is determined by the attProp variable. After the
            //data has been correctly sampled, each tree's buildClassifier method is called, passing the sampled data.
            Instances tempData = new Instances(data);

            int attTracker = data.numAttributes()-1;

            for (int j = 0; j < (int)(data.numAttributes() * (1 - attProp)); j++) {
                Random rand = new Random();

                int attIndex = rand.nextInt(attTracker);

                tempData.deleteAttributeAt(attIndex);

                attTracker--;
            }

            treeEnsemble[i].buildClassifier(tempData);
        }

        //The classDistro is used later in the distributionForInstance method, it is a simple array of double
        //the same size as the number of classes in the dataset.
        classDistro = new double[data.numClasses()];
    }

    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

        double[] classPreds = new double[classDistro.length];

        //An array is made to store the returned prediction from each tree for the class distribution
        for (int i = 0; i < numTrees; i++) {
            //Classify instance is called for each tree in the ensemble, the returned class value is then used to
            //increment the class count array at the correct index
            classPreds[(int) treeEnsemble[i].classifyInstance(instance)]++;
        }

        //once the class count array is complete, the largest value is found and its index is returned to signify the
        //result of the majority vote for the ensemble
        return findLargestVal(classPreds);
    }


    public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {

        classDistro = new double[classDistro.length];

        //The distribution for an isntance can be calculated in two ways, firstly, the distributions from each
        //tree in the ensemble are summed into one array, and this is then averaged to give the average distribution
        //across all of the tree
        if(averageDistributions){
            for (int i = 0; i < numTrees; i++) {
                double[] tempDistro = treeEnsemble[i].distributionForInstance(instance);

                for (int j = 0; j < tempDistro.length; j++) {
                    classDistro[j] += tempDistro[j];
                }
            }
        } else {
            //Secondly, classify instance is called for each tree in the ensemble, the returned predictions are then
            //divided by the number of instances to give the probability of each class being predicted, creating a
            //probability distribution.
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

        Instances[] trainTestSplit = InstanceTools.resampleInstances(optDigitsInstances, 0, 0.7);

        TreeEnsemble optTreeEnsemble = new TreeEnsemble();

        optTreeEnsemble.buildClassifier(trainTestSplit[0]);



        System.out.println("TreeEnsemble on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optTreeEnsemble));

        //for (int i = 0; i < 5; i++) { System.out.println(Arrays.toString(optTreeEnsemble.distributionForInstance(trainTestSplit[1].get(i)))); }

        System.out.println(" ");
        System.out.println(" ");

        /*
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

         */
    }
}
