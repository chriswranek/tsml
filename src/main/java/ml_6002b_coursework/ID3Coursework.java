/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Id3.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.ArrayList;
import java.util.Enumeration;

/**

* Adaptation of the Id3 Weka classifier for use in machine learning coursework (6002B)

 <!-- globalinfo-start -->
 * Class for constructing an unpruned decision tree based on the ID3 algorithm. Can only deal with nominal attributes. No missing values allowed. Empty leaves may result in unclassified instances. For more information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning. 1(1):81-106.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 6404 $ 
 */
public class ID3Coursework
  extends AbstractClassifier 
  implements TechnicalInformationHandler, Sourcable {

  /** for serialization */
  static final long serialVersionUID = -2693678647096322561L;
  
  /** The node's successors. */ 
  private ID3Coursework[] m_Successors;

  /** Attribute used for splitting. */
  private Attribute m_Attribute;

  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution if node is leaf. */
  private double[] m_Distribution;

  private int attOption;

  private boolean discretized = false;

  private int numOfBins = 10;



  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;
  private AttributeSplitMeasure attSplit = new IGAttributeSplitMeasure();

  public void setDiscretized(boolean bool){
    discretized = bool;
  }

  public void setNumOfBins(int num){
    numOfBins = num;
  }

  /** Set options is used to select which method
   * of attribute selection is used when forming the decision tree
   * for the classifier. In treeEnsemble, the selection method
   * for each tree is chosen at random.
   * **/
  public void setOptions(int option){
    switch(option){
      case 0:
        attOption = option;
        attSplit = new IGAttributeSplitMeasure();
        break;
      case 1:
        attOption = option;
        attSplit = new GiniAttributeSplitMeasure();
        break;
      case 2:
        attOption = option;
        attSplit = new ChiSquaredAttributeSplitMeasure();
        break;
      case 3:
        attOption = option;
        IGAttributeSplitMeasure igAtt = new IGAttributeSplitMeasure();
        igAtt.setUseGain(true);
        attSplit = igAtt;
    }
  }

  public String getAttSplit(){
    return attSplit.toString();
  }



  /**
   * Returns a string describing the classifier.
   * @return a description suitable for the GUI.
   */
  public String globalInfo() {

    return  "Class for constructing an unpruned decision tree based on the ID3 "
      + "algorithm. Can only deal with nominal attributes. No missing values "
      + "allowed. Empty leaves may result in unclassified instances. For more "
      + "information see: \n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "R. Quinlan");
    result.setValue(Field.YEAR, "1986");
    result.setValue(Field.TITLE, "Induction of decision trees");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "1");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.PAGES, "81-106");
    
    return result;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  /**
   * Builds Id3 decision tree classifier.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    makeTree(data, attOption);
  }

  /**
   * Method for building an Id3 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, int attOption) throws Exception {

    //System.out.println(data.numAttributes());

    // Check if no instances have reached this node.
    if (data.numInstances() == 0) {
      m_Attribute = null;
      m_ClassValue = Utils.missingValue();
      m_Distribution = new double[data.numClasses()];
      return;
    }

    // Compute attribute with maximum information gain.
    double[] infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()] = attSplit.computeAttributeQuality(data, att);
    }
    m_Attribute = data.attribute(Utils.maxIndex(infoGains));

    
    // Make leaf if information gain is zero. 
    // Otherwise create successors.
    if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
      m_Attribute = null;
      m_Distribution = new double[data.numClasses()];
      Enumeration instEnum = data.enumerateInstances();
      while (instEnum.hasMoreElements()) {
        Instance inst = (Instance) instEnum.nextElement();
        m_Distribution[(int) inst.classValue()]++;
      }
      Utils.normalize(m_Distribution);
      m_ClassValue = Utils.maxIndex(m_Distribution);
      m_ClassAttribute = data.classAttribute();
    } else {
      Instances[] splitData;

      //This section handles the data splitting for the classifier based on the data being nominal or numeric, and then
      //if numeric whether the data is discretized or not. If the data is not discretized, then a binary data split is
      //performed and two successor nodes are created. If the data is discretized using the discretize data method, then
      //the data is split into 10 bins, so each node requires 10 successors to it since each attribute will now have
      //10 discrete values
      if(m_Attribute.isNumeric() && !discretized){
        splitData = attSplit.splitDataOnNumeric(data, m_Attribute, 0, false, 0);

        m_Successors = new ID3Coursework[2];

        for (int j = 0; j < 2; j++) {
          m_Successors[j] = new ID3Coursework();
          m_Successors[j].setOptions(attOption);
          m_Successors[j].setDiscretized(false);
          m_Successors[j].makeTree(splitData[j], attOption);
        }

      } else if(m_Attribute.isNumeric() && discretized){
        splitData = attSplit.splitDataOnNumeric(data, m_Attribute, 0, true, numOfBins);

        m_Successors = new ID3Coursework[numOfBins];

        for (int j = 0; j < numOfBins; j++) {
          m_Successors[j] = new ID3Coursework();
          m_Successors[j].setOptions(attOption);
          m_Successors[j].setDiscretized(true);
          m_Successors[j].makeTree(splitData[j], attOption);
        }
      } else {
        splitData = attSplit.splitData(data, m_Attribute);

        m_Successors = new ID3Coursework[m_Attribute.numValues()];

        for (int j = 0; j < m_Attribute.numValues(); j++) {
          m_Successors[j] = new ID3Coursework();
          m_Successors[j].setOptions(attOption);
          m_Successors[j].makeTree(splitData[j], attOption);
        }
      }

    }
  }

  /**
   * Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                   + "please.");
    }
    if (m_Attribute == null) {
      return m_ClassValue;
    } else {

      return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);
    }
  }

  /**
   * Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double[] distributionForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                   + "please.");
    }
    if (m_Attribute == null) {
      return m_Distribution;
    } else { 
      return m_Successors[(int) instance.value(m_Attribute)].
        distributionForInstance(instance);
    }
  }

  /**
   * Prints the decision tree using the private toString method from below.
   *
   * @return a textual description of the classifier
   */
  public String toString() {

    if ((m_Distribution == null) && (m_Successors == null)) {
      return "Id3: No model built yet.";
    }
    return "Id3\n\n" + toString(0);
  }


  /**
   * Outputs a tree at a certain level.
   *
   * @param level the level at which the tree is to be printed
   * @return the tree as string at the given level
   */
  private String toString(int level) {

    StringBuffer text = new StringBuffer();
    
    if (m_Attribute == null) {
      if (Utils.isMissingValue(m_ClassValue)) {
        text.append(": null");
      } else {
        text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
      } 
    } else {
      for (int j = 0; j < m_Attribute.numValues(); j++) {
        text.append("\n");
        for (int i = 0; i < level; i++) {
          text.append("|  ");
        }
        text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
        text.append(m_Successors[j].toString(level + 1));
      }
    }
    return text.toString();
  }

  /**
   * Adds this tree recursively to the buffer.
   * 
   * @param id          the unqiue id for the method
   * @param buffer      the buffer to add the source code to
   * @return            the last ID being used
   * @throws Exception  if something goes wrong
   */
  protected int toSource(int id, StringBuffer buffer) throws Exception {
    int                 result;
    int                 i;
    int                 newID;
    StringBuffer[]      subBuffers;
    
    buffer.append("\n");
    buffer.append("  protected static double node" + id + "(Object[] i) {\n");
    
    // leaf?
    if (m_Attribute == null) {
      result = id;
      if (Double.isNaN(m_ClassValue)) {
        buffer.append("    return Double.NaN;");
      } else {
        buffer.append("    return " + m_ClassValue + ";");
      }
      if (m_ClassAttribute != null) {
        buffer.append(" // " + m_ClassAttribute.value((int) m_ClassValue));
      }
      buffer.append("\n");
      buffer.append("  }\n");
    } else {
      buffer.append("    checkMissing(i, " + m_Attribute.index() + ");\n\n");
      buffer.append("    // " + m_Attribute.name() + "\n");
      
      // subtree calls
      subBuffers = new StringBuffer[m_Attribute.numValues()];
      newID = id;
      for (i = 0; i < m_Attribute.numValues(); i++) {
        newID++;

        buffer.append("    ");
        if (i > 0) {
          buffer.append("else ");
        }
        buffer.append("if (((String) i[" + m_Attribute.index() 
            + "]).equals(\"" + m_Attribute.value(i) + "\"))\n");
        buffer.append("      return node" + newID + "(i);\n");

        subBuffers[i] = new StringBuffer();
        newID = m_Successors[i].toSource(newID, subBuffers[i]);
      }
      buffer.append("    else\n");
      buffer.append("      throw new IllegalArgumentException(\"Value '\" + i["
          + m_Attribute.index() + "] + \"' is not allowed!\");\n");
      buffer.append("  }\n");

      // output subtree code
      for (i = 0; i < m_Attribute.numValues(); i++) {
        buffer.append(subBuffers[i].toString());
      }
      subBuffers = null;
      
      result = newID;
    }
    
    return result;
  }
  
  /**
   * Returns a string that describes the classifier as source. The
   * classifier will be contained in a class with the given name (there may
   * be auxiliary classes),
   * and will contain a method with the signature:
   * <pre><code>
   * public static double classify(Object[] i);
   * </code></pre>
   * where the array <code>i</code> contains elements that are either
   * Double, String, with missing values represented as null. The generated
   * code is public domain and comes with no warranty. <br/>
   * Note: works only if class attribute is the last attribute in the dataset.
   *
   * @param className the name that should be given to the source class.
   * @return the object source described by a string
   * @throws Exception if the source can't be computed
   */
  public String toSource(String className) throws Exception {
    StringBuffer        result;
    int                 id;
    
    result = new StringBuffer();

    result.append("class " + className + " {\n");
    result.append("  private static void checkMissing(Object[] i, int index) {\n");
    result.append("    if (i[index] == null)\n");
    result.append("      throw new IllegalArgumentException(\"Null values "
        + "are not allowed!\");\n");
    result.append("  }\n\n");
    result.append("  public static double classify(Object[] i) {\n");
    id = 0;
    result.append("    return node" + id + "(i);\n");
    result.append("  }\n");
    toSource(id, result);
    result.append("}\n");

    return result.toString();
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 6404 $");
  }

  /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) throws Exception {


    String optDigitsDataset = "src\\main\\java\\ml_6002b_coursework\\test_data\\optdigits.arff";

    Instances optDigitsInstances = DatasetLoading.loadData(optDigitsDataset);

    Instances[] trainTestSplit = InstanceTools.resampleInstances(optDigitsInstances, 0, Math.random());

    ID3Coursework optIGClassifier = new ID3Coursework();
    optIGClassifier.setOptions(0);

    optIGClassifier.buildClassifier(trainTestSplit[0]);

    System.out.println("DT using measure Information Gain on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optIGClassifier));


    ID3Coursework optGiniClassifier = new ID3Coursework();
    optGiniClassifier.setOptions(1);

    optGiniClassifier.buildClassifier(trainTestSplit[0]);

    System.out.println("DT using measure Gini on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optGiniClassifier));

    ID3Coursework optChiClassifier = new ID3Coursework();
    optChiClassifier.setOptions(2);

    optChiClassifier.buildClassifier(trainTestSplit[0]);

    System.out.println("DT using measure Chi Squared on optdigits problem has test accuracy = " + ClassifierTools.accuracy(trainTestSplit[1], optChiClassifier));


    System.out.println(" ");
    System.out.println(" ");


    String chinaTownDatasetTrain = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TRAIN.arff";
    String chinaTownDatasetTest = "src\\main\\java\\ml_6002b_coursework\\test_data\\ChinaTown_TEST.arff";

    Instances chinaTownTrain = DatasetLoading.loadData(chinaTownDatasetTrain);
    Instances chinaTownTest = DatasetLoading.loadData(chinaTownDatasetTest);

    //The discretizeDataset method is a method that I created and added to the Discretize class in Weka
    //It takes in a numeric dataset of any size and partitions the attributes into 10 bins. Once all the values and
    //instances have been discretized, the dataset is returned and is able to be used in nominal classifiers.
    Instances discretizedChinaTownTrain = Discretize.discretizeDataset(chinaTownTrain, 10);
    Instances discretizedChinaTownTest  = Discretize.discretizeDataset(chinaTownTest, 10);


    //For numeric  data, further parameters need to be set up so that the classifier can handle the data correctly
    //The classifier has 3 different options for handling data, firstly if the data has been discretized into bins
    //then the discretized variable must be set to true so the classifier builds enough successors. If the data is
    //numeric but not discretized, then a binary split is performed on the data at each tree node. If the data is
    //nominal then it is handled normally by the classifier.
    ID3Coursework IGClassifier = new ID3Coursework();
    IGClassifier.setOptions(0);
    IGClassifier.setDiscretized(true);
    IGClassifier.setNumOfBins(10);

    IGClassifier.buildClassifier(discretizedChinaTownTrain);

    System.out.println("DT using measure Information Gain on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(discretizedChinaTownTest, IGClassifier));

    ID3Coursework giniClassifier = new ID3Coursework();
    giniClassifier.setOptions(1);
    giniClassifier.setDiscretized(true);
    giniClassifier.setNumOfBins(10);

    giniClassifier.buildClassifier(discretizedChinaTownTrain);

    System.out.println("DT using measure Gini on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(discretizedChinaTownTest, giniClassifier));

    ID3Coursework chiClassifier = new ID3Coursework();
    chiClassifier.setOptions(2);
    chiClassifier.setDiscretized(true);
    chiClassifier.setNumOfBins(10);

    chiClassifier.buildClassifier(discretizedChinaTownTrain);

    System.out.println("DT using measure Chi Squared on ChinaTown problem has test accuracy = " + ClassifierTools.accuracy(discretizedChinaTownTest, chiClassifier));


  }
}
