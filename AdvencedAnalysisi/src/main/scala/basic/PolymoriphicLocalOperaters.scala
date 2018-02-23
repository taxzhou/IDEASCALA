package basic

object PolymoriphicLocalOperaters {
  def listOfDuplicates[A](x:A,length:Int):List[A] =
    if(length < 1)
      Nil
    else {
      x :: listOfDuplicates(x, length-1)
    }
  
  val x = 1+2*3
  val y = x.toString()
  def succ(x:Int) =x + 1
   
  def fac(n:Int): Int = if (n==0) 1 else n * fac(n-1)
  
  case class MyPair[A,B](x:A,y:B)
  
  def id[T](x:T) = x
  val p = MyPair(1,"scala")
  val q = id(1) 
  
  def main(args: Array[String]): Unit = {
    println(listOfDuplicates[Int](3, 4))
    println(listOfDuplicates[String]("La", 8))
    println(q)
    println(p)
        
  }
}