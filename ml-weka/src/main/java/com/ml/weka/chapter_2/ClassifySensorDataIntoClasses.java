package com.ml.weka.chapter_2;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


public class ClassifySensorDataIntoClasses {
  public static void main(String[] args) throws Exception {
    Instances trainData = loadTrainData();

    NaiveBayes model = buildBayesClassifier(trainData);

    Instances testData = loadTestDataForCrossValidation();

    validateByAssigningToClasses(model, testData);


  }

  private static void validateByAssigningToClasses(NaiveBayes nb, Instances testData) throws Exception {
    for (int j = 0; j < testData.numInstances(); j++) {
      String actual = getClassForTestData(testData, j);
      Instance newInst = testData.instance(j);
      System.out.println("Actual class from test dataSet:" + newInst.stringValue(newInst.numAttributes() - 1));
      String predictedClass = predictClassUsingModel(nb, testData, newInst);
      System.out.println("for sensor data: " + testData.instance(j) + " predicted that it will fall to the class: " + predictedClass);
    }
  }

  private static String predictClassUsingModel(NaiveBayes nb, Instances testData, Instance newInst) throws Exception {
    double preNB = nb.classifyInstance(newInst);
    return testData.classAttribute().value((int) preNB);
  }

  private static String getClassForTestData(Instances testData, int j) {
    double actualClass = testData.instance(j).classValue();
    return testData.classAttribute().value((int) actualClass);
  }

  private static Instances loadTestDataForCrossValidation() throws Exception {
    DataSource source2 = new DataSource(ClassifySensorDataIntoClasses.class.getClassLoader().getResource("chapter_2/iris-non-labeled.arff").getPath());
    Instances testdata = source2.getDataSet();
    testdata.setClassIndex(extractLastFeatureAsLabel(testdata));
    return testdata;
  }

  private static NaiveBayes buildBayesClassifier(Instances trainData) throws Exception {
    NaiveBayes nb = new NaiveBayes();
    nb.buildClassifier(trainData);
    return nb;
  }

  private static Instances loadTrainData() throws Exception {
    String path = ClassifySensorDataIntoClasses.class.getClassLoader().getResource("chapter_2/iris.arff").getPath();
    DataSource source = new DataSource(path);
    Instances trainData = source.getDataSet();
    System.out.println("number of features from sensors: " + trainData.numAttributes());

    trainData.setClassIndex(extractLastFeatureAsLabel(trainData));

    int numClasses = trainData.numClasses();

    for (int i = 0; i < numClasses; i++) {
      String classLabel = trainData.classAttribute().value(i);
      System.out.println(i + " - class label: " + classLabel);
    }
    return trainData;
  }

  private static int extractLastFeatureAsLabel(Instances trainData) {
    return trainData.numAttributes() - 1;
  }

}
