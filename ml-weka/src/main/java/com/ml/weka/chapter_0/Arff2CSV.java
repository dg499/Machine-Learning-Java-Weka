package com.ml.weka.chapter_0;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

public class Arff2CSV {

  public static void main(String[] args) {
    try {
      ArffLoader loader = new ArffLoader();
      loader.setSource(new File("src/main/resources/chapter_0/weather.arff"));
      Instances data = loader.getDataSet();
      System.out.println(data.toSummaryString());

      CSVSaver saver = new CSVSaver();
      saver.setInstances(data);
      saver.setFile(new File("src/main/resources/chapter_0/weather1.csv"));
      saver.writeBatch();
    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

}
