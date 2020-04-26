package com.ml.weka.chapter_0;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Datasets {

  public static void main(String[] args) {
    try {
      DataSource source2 = new DataSource("src/main/resources/chapter_0/weather.arff");
      Instances testdata = source2.getDataSet();
      
      System.out.println(testdata.toSummaryString());
      
      ArffSaver as = new ArffSaver();
      as.setInstances(testdata);
      as.setFile(new File("src/main/resources/chapter_0/weather1.arff"));
      as.writeBatch();
    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

}
