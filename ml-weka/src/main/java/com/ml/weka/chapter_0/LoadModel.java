package com.ml.weka.chapter_0;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadModel {

  public static void main(String[] args) throws Exception {
    J48 tree = (J48) SerializationHelper.read("src/main/resources/chapter_0/myDT.model");

    DataSource src1 = new DataSource("src/main/resources/chapter_0/segment-test.arff");
    Instances tdt = src1.getDataSet();
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
  }

}

