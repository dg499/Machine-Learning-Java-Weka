# Java Weka Standalone CS634 Data Mining Final Projects Option 1 Course Work
This is the repository for performing classification, pattern recognition, building, saving<br/> and evaluating the model prediction using weka libary.

## Instructions
below maven dependency is a custom built dependency for windows10 platform by following weka collective-classification wiki portal,<br/>
jar is included in the source code, to install this jar locally, post cloning navigate to the directory and execute the following command.

```
mvn install:install-file -DlocalRepositoryPath=repository -DcreateChecksum=true -Dpackaging=jar -Dfile=collective-classification-2019.8.7.jar
 -DgroupId=nz.ac.waikato.cms.weka -DartifactId=collective-classification -Dversion=2019-08-07

if in case the repository folder is created in current directory, copy the repository to the local maven .m2 directory.
C:\Users\<profile>\.m2\repository\nz\ac\waikato\cms\weka
```
 
```
        <dependency>
            <groupId>nz.ac.waikato.cms.weka</groupId>
            <artifactId>collective-classification</artifactId>
            <version>2019-08-07</version>
        </dependency>
```
		
### Technical Requirements

This course has the following software requirements:<br/>
This course has the following software requirements:
	•	IntelliJ IDEA
	•	Java JDK 8
	•	Maven

## Learning Materials & References

I have followed below video tutorials for learning the basics of the weka library.

* [Introduction to Artificial Intelligence with Java [Video]](https://www.packtpub.com/big-data-and-business-intelligence/introduction-artificial-intelligence-java-video)

* [Machine Learning Projects with Java [Video]](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-projects-java-video)


