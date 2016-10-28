import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by boris on 16/10/10.
 */
class origprocess {

}
object origprocess{
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      sparkConf.setAppName("adfdsf")
        sparkConf.setMaster("local[10]") //"yarn","StandLong"
    val sc = new SparkContext(sparkConf)


//    val file: RDD[String] = sc.textFile("file:///Users/boris/Downloads/IF*.gz")
//    val take: Array[String] = file.take(100)
//    take.foreach(a=>println(a))
//    val count = file.count
//    println(count)
//    file.repartition(200).saveAsTextFile("/ws/terminal/newre",classOf[GzipCodec])
//
//
//    val dpidata = sc.textFile("/ws/dpidata/if4_03/province=beijing/pt_time=201610102100").filter(a=>a.split("\\|",-1).length == 38)
//    val register = sc.textFile("/ws/terminal/register/*")
//    val dpimap = dpidata.map(a=>((a.split("\\|")(0),a.split("\\|")(1)),a.split("\\|")(26)))
//    val map = register.map(a=>((a.split("\\|")(2),a.split("\\|")(1)),(a.split("\\|")(5),a.split("\\|")(6))))
//
//    val join = dpimap.join(map)
//    val map1  = join.map(a=>a._1._1+"|"+a._1._2+"|"+a._2._1+"|"+a._2._2._1+"|"+a._2._2._2)
//
//    map1.repartition(10).saveAsTextFile("/ws/terminal/newre3",classOf[GzipCodec])

    val file: RDD[String] = sc.textFile("/Users/boris/DevOps/data/newre3/pa*")
    file.map(a=>(a.split("\\|")(0),1)).reduceByKey((a1,a2)=>a1+a2).map(a=>a._2).repartition(1).saveAsTextFile("/Users/boris/DevOps/data/newre3/value111")
    val collect: Array[String] = file.collect()
    file.take(100)
    println(file.first())
    println(file.count)
//    460030776225095|18964362997|OneTravel/4.4.8.1609241929 CFNetwork/758.5.3 Darwin/15.6.0|Apple|ACM-A5
//     file.map(a=>((a.split("\\|")(3),a.split("\\|")(4),a.split("\\|")(2)),1)).reduceByKey(_+_).sortByKey(true).foreach(a=>println(a))


//
//    join.filter(a=>a._2._2.isEmpty).count
  }

}