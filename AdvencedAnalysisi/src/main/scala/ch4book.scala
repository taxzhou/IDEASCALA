import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

import scala.util.Random

object ch4book {
  def main(args: Array[String]): Unit = {
    tools.loggerLevel.setDefaultLoggerLevel()
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val dataWithoutHeader = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("D:\\dataset\\covtype\\covtype.txt")

    //设置参数的列名称
    val colNames = Seq(
      "Elevation","Aspect","Slope",
      "Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology",
      "Horizontal_Distance_To_Roadways",
      "Hillshade_9am","Hillshade_Noon","Hillshade_3pm",
      "Horizontal_Distance_To_Fire_Points"
    ) ++ (
      (0 until 4).map(i => s"Wilderness_Area_$i")
    ) ++ (
      (0 until 40).map(i => s"Soil_Type_$i")
    ) ++ Seq("Cover_Type")


    //转换data
    val data = dataWithoutHeader.toDF(colNames:_*).
      withColumn("Cover_Type",$"Cover_Type".cast("double"))

    data.show()
    data.head

    val Array(trainData, testData) = data.randomSplit(Array(0.9,0.1))
    trainData.cache()
    testData.cache()


    val runRDF = new RunRDF(spark)

//    runRDF.simpleDecisionTree(trainData ,testData)
//    runRDF.randomClassifier(trainData, testData)
//    runRDF.evaluate(trainData , testData)
//    runRDF.evalueCategorical(trainData, testData)
    runRDF.evaluateForeast(trainData , testData)

  }

}

class RunRDF(private val spark: SparkSession) {

  import spark.implicits._

  def  simpleDecisionTree(trainData: DataFrame, testData : DataFrame): Unit = {

    //将最后一列Cover_Type设置为比对项
    val inputCols = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val assembledTrainData = assembler.transform(trainData)
    println("0----------------------0")
    assembledTrainData.select("featureVector").show(truncate = false)

    //设置模型的参数
    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val model = classifier.fit(assembledTrainData)
    println("1-----------------------1")
    println(model.toDebugString)


    //根据模型的推导结果，找出各个属性对应的模型的重要程度
    println("2----------------------2")
    model.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println)


    //推测值
    val predictions = model.transform(assembledTrainData)

    println("3----------------------3")
    predictions.select("Cover_Type","prediction","probability").
      show(truncate = false)

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction")

    println("4----------------------4")
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val f1 = evaluator.setMetricName("f1").evaluate(predictions)
    println(accuracy)
    println(f1)

    //根据预测结果和实际结果，生成预测混淆矩阵
    val predictionRDD = predictions.
      select("prediction", "Cover_Type").
      as[(Double,Double)].rdd
    val multiclassMetrics = new MulticlassMetrics(predictionRDD)
    println("5----------------------5")
    println(multiclassMetrics.confusionMatrix)


    //生成1-7这7个值的混淆矩阵结果
    val confusionMatrix = predictions.
      groupBy("Cover_Type").
      pivot("prediction", (1 to 7)).
      count().
      na.fill(0.0).
      orderBy("Cover_Type")

    println("6----------------------6")
    confusionMatrix.show()
  }



  def classProbabilities(data: DataFrame): Array[Double] = {
    val total = data.count()
    data.groupBy("Cover_Type").count().
      orderBy("Cover_Type").
      select("count").as[Double].
      map(_ / total).
      collect()
  }

  def randomClassifier(trainData: DataFrame, testData: DataFrame): Unit = {
    val trainPriorProbabilities = classProbabilities(trainData)
    val testPriorProbabilities = classProbabilities(testData)
    val accuracy = trainPriorProbabilities.zip(testPriorProbabilities).map {
      case (trainProb, cvProb) => trainProb * cvProb
    }.sum
    println(accuracy)
  }

  def evaluate(trainData: DataFrame, testData: DataFrame): Unit = {
    val inputCols = trainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("featureVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler,classifier))

    //用来构建决策树的参数
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini","entropy")). //熵或者gini
      addGrid(classifier.maxDepth, Seq(1,20)).
      addGrid(classifier.maxBins, Seq(40,200)).
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      build()


    //构建一个分类器，比对真实值为Cover_Type,设置预测值为prediction，矩阵名为accuracy
    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")


    //设置训练模型的其他参数
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)


    //训练模型
    val validatorModel = validator.fit(trainData)

    val paramsAndMetrics = validatorModel.validationMetrics.
      zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

    //打印处矩阵和参数
    paramsAndMetrics.foreach { case (metric, params) =>
        println(metric)
        println(params)
        println()
    }


    //找出决策树中最好的模型
    val bestModel = validatorModel.bestModel

    println("-------------worst model------------------")
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    println("-------------max validationMetrics ------------------")
    println(validatorModel.validationMetrics.max)

    println("-------------test accuracy------------------")
    val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
    println(testAccuracy)

    println("-------------train accuracy------------------")
    val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
    println(trainAccuracy)
  }

  def unencodeOneHost(data: DataFrame): DataFrame = {
    //读取wilderness数据
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    //设置wilderness的参数
    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")

    val unhotUDF = udf((vec:Vector) => vec.toArray.indexOf(1.0).toDouble)

    // 将wilderness当做预测值时，需要重新编辑数据集
    // 数据集去除掉wilderness，然后通过withColumn添加预测值
    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols:_*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    //读取soil参数
    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    //设置soil参数
    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    // 将soil当做预测值时，需要重新编辑数据集
    // 数据集去除掉soil，然后通过soil添加预测值
    soilAssembler.transform(withWilderness).
      drop(soilCols:_*).
      withColumn("soil", unhotUDF($"soil"))

  }


  //评估分类值的决策树
  def evalueCategorical(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHost(trainData)
    val unencTestData = unencodeOneHost(testData)

    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
    val assembler = new VectorAssembler(). //转换
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer(). // 转换
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    //设置分类的分类器
    val classifier = new DecisionTreeClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(assembler,indexer,classifier))

    //设置分类器的各种配置参数用于优化
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.impurity, Seq("gini","entropy")).    //使用熵或者是使用gini
      addGrid(classifier.maxDepth, Seq(1,20)).
      addGrid(classifier.maxBins, Seq(40,300)).
      addGrid(classifier.minInfoGain,Seq(0.0,0.05)).
      build()

    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    //填入参数到validator用于生成模型
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    val bestModel = validatorModel.bestModel

    println("-----------最好的模型----------------------")
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println("测试模型的正确率：")
    println(testAccuracy)

  }

  def evaluateForeast(trainData: DataFrame, testData: DataFrame): Unit = {
    val unencTrainData = unencodeOneHost(trainData) //生成数据并去产生soil和wilder为target的数据集合
    val unencTestData  = unencodeOneHost(testData)

    val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
    // 属性转换器，将多属性值转换为一个单一值
    val assembler = new VectorAssembler().
      setInputCols(inputCols).
      setOutputCol("featureVector")

    val indexer = new VectorIndexer().
      setMaxCategories(40).
      setInputCol("featureVector").
      setOutputCol("indexedVector")

    val classifier = new RandomForestClassifier().
      setSeed(Random.nextLong()).
      setLabelCol("Cover_Type").
      setFeaturesCol("indexedVector").
      setPredictionCol("prediction").
      setImpurity("entropy"). // 设置评判标准为熵
      setMaxDepth(20).        // 设置最大深度为20
      setMaxBins(300)         // 最大同属为300


    // 将数据转换为RDD的一个转换器
    val pipeline = new Pipeline().setStages(Array(assembler,indexer,classifier))

    // 设置Grid的属性值
    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      addGrid(classifier.numTrees, Seq(1, 10)).
      build()

    val multiclassEval = new MulticlassClassificationEvaluator().
      setLabelCol("Cover_Type").
      setPredictionCol("prediction").
      setMetricName("accuracy")

    //填入参数到validator用于生成模型
    val validator = new TrainValidationSplit().
      setSeed(Random.nextLong()).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

    val validatorModel = validator.fit(unencTrainData)

    val bestModel = validatorModel.bestModel

    val forestModel = bestModel.asInstanceOf[PipelineModel].
      stages.last.asInstanceOf[RandomForestClassificationModel]

    println(forestModel.extractParamMap)
    println(forestModel.getNumTrees)
    forestModel.featureImportances.toArray.zip(inputCols).
      sorted.reverse.foreach(println) // 根据feature的重要性进行排序输出

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
    println("---------------正确率测试值-------------------")
    println(testAccuracy)

    println("---------------推测预测值--------------------")
    bestModel.transform(unencTestData.drop("Cover_Type")).select("prediction").show()

  }

}



