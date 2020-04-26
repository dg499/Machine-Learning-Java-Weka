package com.ml.weka.chapter_0;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SaveModel {

  public static void main(String[] args) throws Exception {
    DataSource src = new DataSource("src/main/resources/chapter_0/segment-challenge.arff");
    Instances dt = src.getDataSet();
    // System.out.println(dt.toSummaryString());
    dt.setClassIndex(dt.numAttributes() - 1);

    String[] options = new String[4];
    options[0] = "-C"; // confidence threshold for pruning default 0.25
    options[1] = "0.1";
    options[2] = "-M"; // set max num of instances defult 2
    options[3] = "2";

    J48 tree = new J48();
    tree.setOptions(options);
    tree.buildClassifier(dt);

    weka.core.SerializationHelper.write("src/main/resources/chapter_0/myDT.model", tree);
  }

}
