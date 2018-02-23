package basic

object HigherOrderFunctions {
  def apply(f: Int => String, v: Int) = f(v)
  
  class Decorator(left:String,right:String) {
    def layout[A] (x: A) = left + x.toString() + right
  }
  
  
  def factorial(x: Int):Int = {
    def fact(x:Int, accumulator:Int):Int = {
      if (x <= 1) accumulator
      else fact(x - 1, x * accumulator)
    }
    fact(x, 1)
  }
  
  def filter(xs: List[Int], p: Int => Boolean) : List[Int] =
    if (xs.isEmpty) xs
    else if (p(xs.head)) xs.head :: filter(xs.tail, p)
    else  0 :: filter(xs.tail, p)
  
  def modN(n : Int)(x: Int) = ((x % n) == 0)
  
  
    
  
  def main(args: Array[String]): Unit = {
    val decorator = new Decorator("[","]")
    println(apply(decorator.layout, 7))
    
    println("Factorial of 2 : " + factorial(2))
    println("Factorial of 3 : " + factorial(3))
    
    val nums = List(1,2,3,4,5,6,7,8,9)
    
    println(filter(nums, modN(2)))
    println(filter(nums, modN(3)))
    println(modN(2)(4))
    
    
  }
  
  
  
}
