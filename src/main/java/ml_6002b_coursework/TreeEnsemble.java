package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.RotationForest;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    int numOfBins = 10;
    int numTrees = 50;
    boolean averageDistributions = false;
    ID3Coursework[] treeEnsemble;
    double[] classDistro;
    double attProp = 0.5;
    ArrayList<String> attIndices;
    boolean discretized = false;



    public void setDiscretized(boolean bool){
        discretized = bool;
    }

    public void setNumOfBins(int num){
        numOfBins = num;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        treeEnsemble = new ID3Coursework[numTrees];
        attIndices = new ArrayList<>();

        //This first loop sets the attribute selection method for each of the trees in the ensemble
        //There are 4 methods to choose from, so a random number between 0 and 3 is generated and
        //setOptions is called for each tree, giving it the random number to set its selection method
        int max = 4;
        for (int i = 0; i < numTrees; i++) {
            treeEnsemble[i] = new ID3Coursework();
            Random r = new Random();

            treeEnsemble[i].setOptions(r.nextInt(max));
            treeEnsemble[i].setDiscretized(discretized);
        }


        for (int i = 0; i < numTrees; i++) {

            //For each tree in the ensemble, a new set of instances is generated, and for each of the new instances
            //its attributes are sampled randomly, how many are sampled is determined by the attProp variable. After the
            //data has been correctly sampled, each tree's buildClassifier method is called, passing the sampled data.
            Instances tempData = new Instances(data);
            Instances newData;
            Remove remove = new Remove();

            //System.out.println(tempData);

            if(attProp < 1){
                ArrayList<Integer> arrayList = new ArrayList<>();
                int[] attArr = getAttributeIndices(arrayList, data.numAttributes(), attProp);

                String attributeIndices = Arrays.toString(attArr);
                String test = attributeIndices.replaceAll("[\\[\\](){}]","");
                attIndices.add(test);

                remove.setAttributeIndices(test);
                remove.setInvertSelection(false);
                remove.setInputFormat(tempData);
                newData = Filter.useFilter(tempData, remove);

                //System.out.println(newData);

                treeEnsemble[i].buildClassifier(newData);
            } else {
                treeEnsemble[i].buildClassifier(data);
            }



        }

        //The classDistro is used later in the distributionForInstance method, it is a simple array of double
        //the same size as the number of classes in the dataset.
        classDistro = new double[data.numClasses()];
    }

    public int[] getAttributeIndices(ArrayList<Integer> arrayList, int numOfAttributes, double attProportion){

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

    public double classifyInstance(Instance instance) throws Exception {

        double[] classPreds = new double[classDistro.length];
        Remove remove = new Remove();

        //TRIM INSTANCE TO HAVE ONLY THE SAME ATTRIBUTES AS WERE SELECTED IN THE ENSEMBLE CONSTRUCTION

        //An array is made to store the returned prediction from each tree for the class distribution
        for (int i = 0; i < numTrees; i++) {
            if(attProp < 1){

                //Filters the instance to remove the same attributes as were removed when building the classifier
                //The attIndices arraylist stores which attributes are removed from each classifier in the ensemble
                Instances newData;
                remove.setAttributeIndices(attIndices.get(i));
                remove.setInvertSelection(false);
                remove.setInputFormat(instance.dataset());

                newData = Filter.useFilter(instance.dataset(), remove);

                //System.out.println("  Cleaned Instance: "+ newData.instance(instance.dataset().indexOf(instance)));
                //Classify instance is called for each tree in the ensemble, the returned class value is then used to
                //increment the class count array at the correct index

                //Once the instance has been filtered it can now be classified by the ensemble
                classPreds[(int) treeEnsemble[i].classifyInstance(newData.instance(instance.dataset().indexOf(instance)))]++;
            } else {
                classPreds[(int) treeEnsemble[i].classifyInstance(instance)]++;
            }

        }

        //once the class count array is complete, the largest value is found and its index is returned to signify the
        //result of the majority vote for the ensemble
        return findLargestVal(classPreds);
    }


    public double[] distributionForInstance(Instance instance) throws Exception {

        classDistro = new double[classDistro.length];
        Remove remove = new Remove();
        //The distribution for an instance can be calculated in two ways, firstly, the distributions from each
        //tree in the ensemble are summed into one array, and this is then averaged to give the average distribution
        //across all of the tree
        if(averageDistributions){
            for (int i = 0; i < numTrees; i++) {

                double[] tempDistro;

                if(attProp < 1){
                    Instances newData;
                    remove.setAttributeIndices(attIndices.get(i));
                    remove.setInvertSelection(false);
                    remove.setInputFormat(instance.dataset());

                    newData = Filter.useFilter(instance.dataset(), remove);

                    tempDistro = treeEnsemble[i].distributionForInstance(newData.instance(instance.dataset().indexOf(instance)));

                } else {
                    tempDistro = treeEnsemble[i].distributionForInstance(instance);
                }

                for (int j = 0; j < tempDistro.length; j++) {
                    classDistro[j] += tempDistro[j];
                }
            }
        } else {
            //Secondly, classify instance is called for each tree in the ensemble, the returned predictions are then
            //divided by the number of instances to give the probability of each class being predicted, creating a
            //probability distribution.
            for (int i = 0; i < numTrees; i++) {

                if(attProp < 1){
                    Instances newData;
                    remove.setAttributeIndices(attIndices.get(i));
                    remove.setInvertSelection(false);
                    remove.setInputFormat(instance.dataset());

                    newData = Filter.useFilter(instance.dataset(), remove);

                    classDistro[(int) treeEnsemble[i].classifyInstance(newData.instance(instance.dataset().indexOf(instance)))]++;

                } else {
                    classDistro[(int) treeEnsemble[i].classifyInstance(instance)]++;
                }

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


        for (int i = 0; i < 5; i++) { System.out.println(Arrays.toString(optTreeEnsemble.distributionForInstance(trainTestSplit[1].get(i)))); }

        System.out.println(" ");
        System.out.println(" ");


        String chinaTownDatasetTrain = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TRAIN.arff";
        String chinaTownDatasetTest = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TEST.arff";

        Instances chinaTownTrain = DatasetLoading.loadData(chinaTownDatasetTrain);
        Instances chinaTownTest = DatasetLoading.loadData(chinaTownDatasetTest);

        Instances discretizedChinaTownTrain = Discretize.discretizeDataset(chinaTownTrain, 10);
        Instances discretizedChinaTownTest  = Discretize.discretizeDataset(chinaTownTest, 10);

        TreeEnsemble chinaEnsemble = new TreeEnsemble();
        chinaEnsemble.setDiscretized(true);
        chinaEnsemble.setNumOfBins(10);

        chinaEnsemble.buildClassifier(discretizedChinaTownTrain);

        System.out.println("TreeEnsemble on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(discretizedChinaTownTest, chinaEnsemble));

        for (int i = 0; i < 5; i++) {
            System.out.println(Arrays.toString(chinaEnsemble.distributionForInstance(discretizedChinaTownTest.get(i))));
        }



    }
}
