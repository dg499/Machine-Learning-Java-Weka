package com.ml.weka.chapter_0.clustering;

import weka.classifiers.collective.functions.LLGC;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SemiSuperClassifier {

  public static void main(String[] args) throws Exception {

    DataSource source = new DataSource("src/main/resources/chapter_0/weather.arff");
    Instances data = source.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);

    LLGC model = new LLGC();
    model.buildClassifier(data);
    System.out.println(model.getCapabilities());
  }
}
