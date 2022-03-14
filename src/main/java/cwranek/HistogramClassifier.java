package cwranek;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class HistogramClassifier extends AbstractClassifier {
    int numOfBins = 10;
    int defaultAttIndex = 0;

    int femaleMean;
    int maleMean;
    //int drawMean;

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        instances = new Instances(instances);

        if(instances.numClasses() > 3){
            throw new Exception("Can only handle one class(es)");
        }


        int fCount=0, mCount=0;

        for (Instance ins: instances) {
            //ins.classValue means a win or a loss
            //System.out.println(ins.classValue());

            if(ins.classValue() == 0){
                maleMean += ins.value(0);
                mCount++;
            } else {
                femaleMean += ins.value(0);
                fCount++;
            }
        }

        maleMean/=mCount;
        femaleMean/=fCount;

        //System.out.println(fCount);
        //System.out.println(drawCount);
        //System.out.println(mCount);

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = this.distributionForInstance(instance);

        if(dist == null){
            throw new Exception("Null distribution predicted");
        } else {
            switch(instance.classAttribute().type()){
                case 0:
                case 3:
                    return dist[0];
                case 1:
                    double max = 0.0;
                    int maxIndex = 0;

                    for (int i = 0; i < dist.length; ++i) {
                        if(dist[i] > max){
                            maxIndex = i;
                            max = dist[i];
                        }
                    }

                    if(max > 0.0){
                        return (double)maxIndex;
                    }

                    return Utils.missingValue();
                case 2:
                default:
                    return Utils.missingValue();
            }
        }

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] probability = new double[instance.numClasses()];
        double x = instance.value(0);
        double distToMaleMean = Math.abs(x-maleMean);
        //double distToDrawMean = Math.abs(x-drawMean);
        double distToFemaleMean = Math.abs(x-femaleMean);

        probability[0] = distToFemaleMean/(distToMaleMean+distToFemaleMean);
        probability[1] = distToMaleMean/(distToMaleMean+distToFemaleMean);
        //probability[2] = distToNegMean/(distToNegMean+distToDrawMean+distToPosMean);

        return probability;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    public void setNumOfBins(int numOfBins){
        this.numOfBins = numOfBins;
    }

    public void setDefaultAttIndex(int defaultAttIndex) {
        this.defaultAttIndex = defaultAttIndex;
    }
}
