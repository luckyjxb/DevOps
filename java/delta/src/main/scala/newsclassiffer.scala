import java.io.StringReader

import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

//import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.wltea.analyzer.lucene.IKAnalyzer

/**
 * Created by boris on 16/12/13.
 */
/**
 * 自定义case类型
 * @param category 分类
 * @param text 新闻正文
 */
case class RawDataRecord(category: String, text: String)

/**
 * 基于贝叶斯的新闻分类器
 * by spark-1.6.0
 */
object newsclassifferByNaiveBayes {

  def main(args: Array[String]) {
    //训练文件
    var trainFile = "/Users/boris/Downloads/dianxin03_news_data/t1.txt"
    //spark master 资源
    var master = "local[30]"
    //spark应用名称
    var appName = "delta_bayes"

    /**
     * 从命令行传入训练集文件和master
     * 用于在大数据集群上跑使用
     */
    if(args.length>=2){
      master = args(1)
    }
    if(args.length>=1){
      trainFile = args(0)
    }
//    val spark = SparkSession.builder.master(master ).appName("delta_Bayes").getOrCreate()

    //spark初使化
    val conf = new SparkConf().setMaster(master).setAppName(appName)
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    /**
     * 读取训练文件，通过IKAnalyzer进行分词
     * 数据组成RawDataRecord(分类，"词1 词2 词3 。。。。")类型
     * 分类类型的数据值转换偷懒了，未使用字典，只使用了hascode生成一个数据，见79行
     */

    val srcRDD = sc.textFile(trainFile).map {
      x =>
        var data = x.split("\t")
        val analyzer = new IKAnalyzer();
        val tokenStream = analyzer.tokenStream("content",
          new StringReader(data(0)))

        tokenStream.getAttribute(classOf[CharTermAttribute])
        tokenStream.reset()
        var result: List[String] = List()
        var pingS = ""
        while (tokenStream.incrementToken()) {
          val charTermAttribute = tokenStream
            .getAttribute(classOf[CharTermAttribute])
          pingS = pingS + "\t" + charTermAttribute.toString
        }
        val join: String = pingS // result.mkString("\t")
        RawDataRecord(data(1).hashCode.toString(), join)
    }
    //以下注释代码为打印调试数据，需要可以打开，但会影响运行速度
    /*
    srcRDD.cache
    srcRDD.toDF.show(100)
    val take1: Array[RawDataRecord] = srcRDD.take(2)
    println("map " + take1(1))
    */



    //对训练样本文件随机划分，按9：1划分成训练集和测试集
    val splits = srcRDD.randomSplit(Array(0.9, 0.1))
    //创建训练集DataFrame和测试集DataFrame
    var trainingDF = splits(0).toDF()
    var testDF = splits(1).toDF()
    //使用标记器对训练集的text（见第一步的map过程，创建了一个RawDataRecord对象，里面有一个属性）进行转换，转换结果放在words字段中
    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(trainingDF)
//    println("output1：")
//    val take: Array[Row] = wordsData.select($"category", $"text", $"words").take(1)
//
//    println(take(0))

    //统计词频特征，对words字段进行转换，生成rawFeatures字段
    var hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    var featurizedData = hashingTF.transform(wordsData)
//    println("output2：")
//    val p2 = featurizedData.select($"category", $"words", $"rawFeatures").take(1)
//    print(p2(0))

    //计算每个词的TF-IDF
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idfModel = idf.fit(featurizedData)
    var rescaledData = idfModel.transform(featurizedData)
//    println("output3：")
//    val take2: Array[Row] = rescaledData.select($"category", $"features").take(1)
    //经过特征转换后，把类型与特征码两段取出来，将特征向量抽象为 LabeldPoint
    val trainDataRdd = rescaledData.select($"category",$"features").map {
      case Row(label: String, features: SparseVector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
//    println("output4：")
//    trainDataRdd.take(1)

    //开始训练模型，使用贝叶斯算法
    val model = NaiveBayes.train(trainDataRdd, lambda = 1.0, modelType = "multinomial")
    //测试数据集，做同样的特征表示及格式转换，与前面训练集转换过程相同
    var testwordsData = tokenizer.transform(testDF)
    var testfeaturizedData = hashingTF.transform(testwordsData)
    var testrescaledData = idfModel.transform(testfeaturizedData)
    var testDataRdd = testrescaledData.select($"category", $"features").map {
      case Row(label: String, features: SparseVector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
    //对测试数据集使用训练模型进行分类预测
    val testpredictionAndLabel = testDataRdd.map(p => (model.predict(p.features), p.label))
    //统计分类准确率
    var testaccuracy = 1.0 * testpredictionAndLabel.filter(x => x._1 == x._2).count() / testDataRdd.count()
    println("output5：")
    println(testaccuracy)
  }
}




