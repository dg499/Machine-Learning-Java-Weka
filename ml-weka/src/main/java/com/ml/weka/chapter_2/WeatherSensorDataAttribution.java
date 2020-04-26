package com.ml.weka.chapter_2;


import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;


public class WeatherSensorDataAttribution {
  public static void main(String[] args) throws Exception {
    Instances data = loadTestData();

    int numAttr = data.numAttributes() - 1;
    System.out.println("the number of features in sensor data:" + numAttr);

    for (int i = 0; i < numAttr; i++) {
      AttributeStats as = data.attributeStats(i);
      int dc = as.distinctCount;
      System.out.println("Feature at index[" + i + "] has " + dc + " distinct value");
      printInfoAboutNumericFeature(data, i, as);
    }
    getLabelsFromTrainData(data);
  }

  private static void getLabelsFromTrainData(Instances data) {
    int numInst = data.numInstances();
    for (int j = 0; j < numInst; j++) {
      Instance instance = data.instance(j);
      double cv = instance.classValue();
      System.out.println(instance.classAttribute().value((int) cv));
    }
  }

  private static void printInfoAboutNumericFeature(Instances data, int i, AttributeStats as) {
    if (data.attribute(i).isNumeric()) {
      System.out.println("Feature at[" + i + "] is numeric");
      Stats s = as.numericStats;
      System.out.println("Feature at[" + i + "] min: " + s.min + " max: " + s.max);
    }
  }

  private static Instances loadTestData() throws Exception {
    String path = WeatherSensorDataAttribution.class.getClassLoader().getResource("chapter_2/weather.arff").getPath();
    DataSource source = new DataSource(path);
    Instances data = source.getDataSet();
    if (data.classIndex() == -1) {
      data.setClassIndex(data.numAttributes() - 1);
    }
    return data;
  }

}
