package basic

import scala.annotation.tailrec

object ParametersAnnotations {
  def calculate(input: => Int) = input*37
  def invode(f: String => Int => Long) = {
    println("call invoke")
    f("1")(2)
  }
  def curry(s: String)(i: Int): Long = {
    s.toLong + i.toLong
  }
  def whileLoop(condition: => Boolean)(body: => Unit) : Unit=
    if(condition) {
      body
      whileLoop(condition)(body)
    }
      
  // call by name parameters will be set values while it executes 
  // end of by name parameters
  
  //Annotations that ensure correctness of encodings
  
  
  def factorial(x: Int) : Int = {
    @tailrec
    def factorialHepler(x : Int, accumulator : Int) : Int = {
      if (x == 1) accumulator 
      else factorialHepler(x - 1, accumulator * x )
    }
    factorialHepler(x, 1)
  }
  
  //default parameter values
  
  def log(message: String, level: String= "Info") = println(s"$level : $message")
  class Point( x : Double = 0,  y: Double = 0) {
    override def toString = s"($x,$y)"
    def printString = {
      println(toString)
    }
  }
  
  // named arguments
  def printName(first:String , last :String) : Unit ={
    println(s"($first,$last)")
  }
  
  
  def main(args: Array[String]) {
    invode{println("eval parameter expression");curry}
    
    var i = 2
    whileLoop(i> 0){
      println(i)
      i -= 1
    }
    
    log("System starting")
    log("USer not found","Warning")
    
    val point1 = new Point(y = 1)
    point1.printString
    
    val point2 = new Point(1)
    point2.printString
    
    printName("John","Simth")
    printName(first = "John", last= "Simth")
    
    
  }
  
}