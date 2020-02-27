package tsml.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface Retrainable {

    boolean isRetrain();

    void setRebuild(boolean state);
}
