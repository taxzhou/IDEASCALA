package basic

import scala.util.matching.Regex
import scala.util.Random

object PatternsExtractorComperhensions {
  val numberPattern: Regex = "[0-9]".r
  val keyValPattern: Regex = "([0-9a-zA-Z-#() ]+): ([0-9a-zA-Z-#() ]+)".r

  val input: String =
    """background-color: #A03300;
      |background-image: url(img/header100.png);
      |background-position: top center;
      |background-repeat: repeat-x;
      |background-size: 2160px 108px;
      |margin: 0;
      |height: 108px;
      |width: 100%;""".stripMargin
      
      
  def apply(name:String) = s"$name---${Random.nextLong()}"
  
  def unapply(customerID:String):Option[String] = {
    val name = customerID.split("-").head
    if(name.nonEmpty) Some(name) else None
  }
  val custorm1ID = PatternsExtractorComperhensions("SunwuKong")
 
  
  case class User(val name:String, val age: Int)
  
  val userBase = List(new User("Travis", 28),
      new User("Kelly", 33),
      new User("Jennifer", 44),
      new User("Dennis", 23))
  
  val twentys = for(user <- userBase if (user.age >=20 && user.age <=30))
    yield user.name
    
  def foo(n:Int,v:Int) = 
    for (i <-0 until n; j <- i until n if i+ j ==v)
      yield (i,j)
    
      
  def main(args: Array[String]): Unit = {
    
    numberPattern.findFirstMatchIn("1awesomepassword") match {
    case Some(_) => println("Passsword OK")
    case None => println("Password must contain a number")
    }
    
    println(input)
    println("________________________")
    for (patternMatch <- keyValPattern.findAllMatchIn(input))
      println(s"key: ${patternMatch.group(1)} value: ${patternMatch.group(2)}")
    println("________________________")
    
    
    custorm1ID match {
      case PatternsExtractorComperhensions(name) => println(name)
      case _ => println("Could not extract a CustomerID")
    } 
    println("________________________")
    twentys.foreach(name => println(name))
    
    foo(10,10) foreach {
      case (i,j) =>
        print(s"($i,$j)")
    }
  }
       
}