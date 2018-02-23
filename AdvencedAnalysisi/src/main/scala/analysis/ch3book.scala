package analysis

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


object ch3book {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()
    spark.sparkContext.setCheckpointDir("/checkpoints")

    val base = "/dataset/profiledata_06-May-2005/"
    val rawUserArtistData = spark.read.textFile(base + "user_artist_data.txt")
    val rawArtistData = spark.read.textFile(base + "artist_data.txt")
    val rawArtistAlias = spark.read.textFile(base + "artist_alias.txt")

    val ch3 = new ch3book(spark)
//    ch3.preparation(rawUserArtistData,rawArtistData,rawArtistAlias)
//    ch3.model(rawUserArtistData,rawArtistData,rawArtistAlias)
    ch3.evluate(rawUserArtistData,rawArtistAlias)
    ch3.recommend(rawUserArtistData,rawArtistData,rawArtistAlias)
  }
}

class ch3book(private val spark: SparkSession) {
  import spark.implicits._

  // 准备数据：
  // rawUserArtistData为所获取到的“用户ID-艺术家ID-count”的稀疏矩阵
  // rawArtistData为artistID和对应的名称
  // rawArtistAlias为artist的正式ID和别名ID
  def preparation(
                   rawUserArtistData : Dataset[String],
                   rawArtistData : Dataset[String],
                   rawArtistAlias : Dataset[String]): Unit = {
    rawArtistData.take(5).foreach(println)
    val userArtistDF = rawUserArtistData.map{line =>
      val Array(user,artist,_*) = line.split(' ')
      (user,artist.toInt)
    }.toDF("user","artist")

    // 查看显示userArtistDF中的最大最小值
    userArtistDF.agg(min("user"), max("user"), min("artist"), max("artist")).show()

    val artistByID = buildArtistByID(rawArtistData)
    val aritstAlias = buildArtistAlias(rawArtistAlias)

    // 提取出artist别名表中的第一行数据
    // 查看显示在artist对应表中，第一行数据对应的两个id只所对应的(aritstID,artistName)
    val (badID,goodID) = aritstAlias.head
    artistByID.filter($"id" isin (badID,goodID)).show()
  }

  // 建立model
  // 使用ALS最小二乘法的方式解析用户-艺术家矩阵
  def model(
             rawUserArtistData : Dataset[String],
             rawArtistData : Dataset[String],
             rawArtistAlias : Dataset[String]): Unit = {
    // 将artistalias数据进行处理，形成map并广播数据
    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

    //生成训练数据集并cache到内存
    val trainData = buildCounts(rawUserArtistData,bArtistAlias).cache()

    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true). // 显示偏好信息-false，隐式偏好信息-true，默认false（显示）
      setRank(10). // 分块数（并行计算）
      setRegParam(0.01). //正则化参数
      setAlpha(1.0). // 只用于隐式的偏好数据，偏好值可信度底
      setMaxIter(5). //最大迭代次数
      setUserCol("user").
      setRatingCol("count").
      setItemCol("artist").
      setPredictionCol("prediction").
      fit(trainData)


    trainData.unpersist()

    model.userFactors.select("features").show(truncate = false)

    val userID = 2093760

    //找到用户2093760的艺术家ID并显示
    val existingArtistIDs = trainData.
      filter($"user" === userID).
      select("artist").as[Int].collect()

    val artistByID = buildArtistByID(rawArtistData)

    // 显示出上述艺术家ID对应的ID，名称
    artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

    // 找出预测头五条并显示
    val topRecommendations = makeRecommendations(model,userID,5)
    topRecommendations.show()

    //头五条显示的数据中对应的艺术家名称
    val recommendArtistIDs = topRecommendations.select("artist").as[Int].collect()

    artistByID.filter($"id" isin (recommendArtistIDs:_*)).show()

    // 清理内存
    model.userFactors.unpersist()
    model.itemFactors.unpersist()
  }


  def evluate(rawUserArtistData: Dataset[String],
              rawArtistAlias: Dataset[String]): Unit = {
    val bAristAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))

    val allData = buildCounts(rawUserArtistData, bAristAlias)

    // 切分数据为9训练数据，1实验数据
    val Array(trainData,cvData) = allData.randomSplit(Array(0.9,0.1))
    trainData.cache()
    cvData.cache()

    // 去除重复的artistID并广播所有数据
    val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
    val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

    // AUC计算出来的结果做基准对比
    // 使用的方法为predictMostListened方法，通过对训练集中艺术家被听歌次数进行简单累加最为预测函数评估
    // 数据为cvData，也就是测试集数据，trainData为训练集
    val mostListenedAUC = areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))
    println(mostListenedAUC)

    val evaluations =
      for(rank <- Seq(5, 30);
          regParam <- Seq(1.0, 0.0001);
          alpha <- Seq(1.0, 40.0))
        yield {
          val model = new ALS().
            setSeed(Random.nextLong()).
            setImplicitPrefs(true).
            setRank(rank).setRegParam(regParam).
            setAlpha(alpha).setMaxIter(20).
            setUserCol("user").setItemCol("artist").
            setRatingCol("count").setPredictionCol("prediction").
            fit(trainData)

          val auc = areaUnderCurve(cvData, bAllArtistIDs, model.transform)

          model.userFactors.unpersist()
          model.itemFactors.unpersist()

          (auc, (rank, regParam, alpha))
        }

    evaluations.sorted.reverse.foreach(println)

    // 清理内存
    trainData.unpersist()
    cvData.unpersist()

  }


  def recommend(
                 rawUserArtistData: Dataset[String],
                 rawArtistData: Dataset[String],
                 rawArtistAlias: Dataset[String]
               ): Unit = {
    val bArtistAlias = spark.sparkContext.broadcast(buildArtistAlias(rawArtistAlias))
    val allData = buildCounts(rawUserArtistData, bArtistAlias).cache()
    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).setRegParam(1.0).
      setAlpha(40.0).setMaxIter(20).
      setUserCol("user").setItemCol("artist").
      setRatingCol("count").setPredictionCol("prediction").
      fit(allData)
    allData.unpersist()

    val userID = 2093760
    val topRecommendations = makeRecommendations(model,userID,5)

    val recommendationArtistIDs = topRecommendations.select("artist").as[Int].collect()
    // 去掉脏数据
    val artistById = buildArtistByID(rawArtistData)

    artistById.join(spark.createDataset(recommendationArtistIDs).toDF("id"),"id").
      select("name").show()


    // 清理内存
    model.userFactors.unpersist()
    model.itemFactors.unpersist()
  }


  // 转换原始的dataset数据为DF数据
  // 去除掉rawArtistData数据中的无名称，错误格式的数据
  def buildArtistByID(rawArtistData: Dataset[String]): DataFrame = {
    rawArtistData.flatMap { line =>
      val(id,name) = line.span(_ != '\t')
      if (name.isEmpty) None
      else {
        try {
          Some((id.toInt,name.trim))
        } catch {
          case e: NumberFormatException => None
        }
      }
    }.toDF("id","name")
  }

  // 转换rawArtistAlias数据从dataset到Map，方便进行后期的查询计算
  // 去除掉错误数据：一行只有一个数值
  def buildArtistAlias(rawArtistAlias: Dataset[String]): Map[Int,Int] = {
    rawArtistAlias.flatMap { line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) None
      else Some((tokens(0).toInt,tokens(1).toInt))
    }.collect().toMap
  }

  // 将rawUserArtistData中的数据转换为DF数据
  // 通过使用广播变量bArtistAlias的map数据，将其中的artist的别名ID全部替换为正确ID
  def buildCounts(rawUserArtistData: Dataset[String],
                  bArtistAlias: Broadcast[Map[Int,Int]]
                 ): DataFrame ={
    rawUserArtistData.map { line =>
      val Array(userID,artistID,count) = line.split(' ').map(_.toInt)
      val finalAritstID = bArtistAlias.value.getOrElse(artistID,artistID)
      (userID,finalAritstID,count)
    }.toDF("user","artist","count")
  }

  // 提取出ALSModel中 对应userID下的howMany个预测值
  def makeRecommendations(model: ALSModel,userID: Int,howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.
      select($"id".as("artist")).
      withColumn("user",lit(userID))


    model.transform(toRecommend).
      select("artist","prediction").
      orderBy($"prediction".desc).
      limit(howMany)
  }

  // AUC算法的实现
  // 返回结果为mean AUC
  def areaUnderCurve(positiveData: DataFrame,
                     bAllArtistIDs: Broadcast[Array[Int]],
                     predictFunction: (DataFrame => DataFrame)) : Double = {

    // Take held-out data as the "positive".
    // Make predictions for each of them, including a numeric score
    val positivePredictions = predictFunction(positiveData.select("user","artist")).
      withColumnRenamed("prediction","positivePrediction")


    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other artists, excluding those that are "positive" for the user.
    val negativeData = positiveData.select("user","artist").as[(Int,Int)].
      groupByKey { case (user, _) => user}.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0

        while(i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))

          // 只接入新的未重复的ID
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // 返回具有user ID的集合
        negative.map(artistID => (userID, artistID))
      }.toDF("user","artist")

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction","negativePrediction")

    // Join positive predictions to negative predictions by user, only.
    // This will result in a row for every possible pairing of positive and negative
    // predictions within each user.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs per user
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user","total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user")).
      select("user","correct")
    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, "user").
      select($"user", ($"correct" / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC

  }

  def predictMostListened(train: DataFrame)(allData: DataFrame): DataFrame = {
    val listenCounts = train.groupBy("artist").
      agg(sum("count").as("prediction")).
      select("artist","prediction")
    allData.
      join(listenCounts,Seq("artist"),"left_outer").
      select("user","artist","prediction")
  }
}
