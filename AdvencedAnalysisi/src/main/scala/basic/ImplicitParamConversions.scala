package basic

import scala.language.implicitConversions

object ImplicitParamConversions {
  abstract class SemiGroup[A] {
    def add(x: A, y:A): A
  }
  abstract class Monoid[A] extends SemiGroup[A] {
    def unit: A
  }
  implicit object StringMonoid extends Monoid[String] {
    def add(x: String,y: String):String =x concat y
    def unit:String =""
  }
    
  implicit object InitMonid extends Monoid[Int] {
    def add(x:Int,y:Int):Int = x+ y
    def unit:Int = 0
  }
  def sum[A](xs:List[A])(implicit m: Monoid[A]):A =
    if(xs.isEmpty) m.unit
    else m.add(xs.head,sum(xs.tail))

  def main(args: Array[String]): Unit = {
    println(sum(List(1, 2, 3)))       // uses IntMonoid implicitly
    println(sum(List("a", "b", "c"))) // uses StringMonoid implicitly
  }
    
}