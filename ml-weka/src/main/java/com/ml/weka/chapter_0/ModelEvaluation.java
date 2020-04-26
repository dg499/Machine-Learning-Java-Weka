package com.ml.weka.chapter_0;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class ModelEvaluation {

  public static void main(String[] args) {
    try {
      DataSource src = new DataSource("src/main/resources/chapter_0/segment-challenge.arff");
      Instances dt = src.getDataSet();
//      System.out.println(dt.toSummaryString());
      dt.setClassIndex(dt.numAttributes() - 1);

      String[] options = new String[4];
      options[0] = "-C"; // confidence threshold for pruning default 0.25
      options[1] = "0.1";
      options[2] = "-M"; // set max num of instances defult 2
      options[3] = "2";

      J48 tree = new J48();
      tree.setOptions(options);
      tree.buildClassifier(dt);

//      System.out.println(tree.getCapabilities().toString());
//      System.out.println(tree.graph());

      Evaluation evaluation = new Evaluation(dt);

      DataSource src1 = new DataSource("src/main/resources/chapter_0/segment-test.arff");
      Instances tdt = src1.getDataSet();
      System.out.println(tdt.toSummaryString());
      tdt.setClassIndex(tdt.numAttributes() - 1);

      evaluation.evaluateModel(tree, tdt);

      System.out.println(evaluation.toSummaryString("Evaluation Results:\n", false));

      System.out.println("Correct % = " + evaluation.pctCorrect());
      System.out.println("Incorrect % = " + evaluation.pctIncorrect());
      System.out.println("kappa = " + evaluation.kappa());
      System.out.println("MAE = " + evaluation.meanAbsoluteError());
      System.out.println("RMSE = " + evaluation.rootMeanSquaredError());
      System.out.println("RAE = " + evaluation.relativeAbsoluteError());
      System.out.println("Precision = " + evaluation.precision(1));
      System.out.println("Recall = " + evaluation.recall(1));
      System.out.println("fMeasure = " + evaluation.fMeasure(1));
      System.out.println("Error Rate = " + evaluation.errorRate());
      System.out.println(evaluation.toMatrixString("Overall Confusion Matrix"));



    } catch (Exception e) {
      System.out.println(e.getMessage());
    }
  }

}
