package com.ml.weka.chapter_0;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CSV2Arff {
  public static void main(String[] args) {
    try {
      CSVLoader loader = new CSVLoader();
      loader.setSource(new File("src/main/resources/chapter_0/weather1.csv"));
      Instances data = loader.getDataSet();
      System.out.println(data.toSummaryString());

      ArffSaver as = new ArffSaver();
      as.setInstances(data);
      as.setFile(new File("src/main/resources/chapter_0/weather1.arff"));
      as.writeBatch();
    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }
}
