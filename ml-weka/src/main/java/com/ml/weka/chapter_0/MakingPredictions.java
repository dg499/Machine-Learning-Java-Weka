package com.ml.weka.chapter_0;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class MakingPredictions {

  public static void main(String[] args) {
    try {
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

      // System.out.println(tree.getCapabilities().toString());
      // System.out.println(tree.graph());



      DataSource src1 = new DataSource("src/main/resources/chapter_0/segment-test.arff");
      Instances tdt = src1.getDataSet();
      System.out.println(tdt.toSummaryString());
      tdt.setClassIndex(tdt.numAttributes() - 1);

      System.out.println("ActualClass \t ActualValue \t PredictedValue \t PredictedClass");

      for (int i = 0; i < tdt.numInstances(); i++) {

        String act = tdt.instance(i).stringValue(tdt.instance(i).numAttributes() - 1);
        double actual = tdt.instance(i).classValue();
        Instance inst = tdt.instance(i);
        double predict = tree.classifyInstance(inst);
        String pred = inst.toString(inst.numAttributes() - 1);

        System.out.println(act + "\t\t" + actual + "\t\t" + predict + "\t\t" + pred);

      }

    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

}
