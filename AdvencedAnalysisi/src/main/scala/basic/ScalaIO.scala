package basic

import java.io._
import scala.io.Source

object ScalaIO {
  def main(args: Array[String]): Unit = {
    val writer = new PrintWriter("D:/test.txt")
    writer.write("Testing for Scala IO, this is using the JAVA IO\n")
    writer.write("Testing for line 2")
    writer.close()
    
    println("____file content is :____")
    Source.fromFile("D:/test.txt") foreach print
  }
}