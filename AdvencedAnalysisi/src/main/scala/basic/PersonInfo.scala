package basic

object PersonInfo {
  class Pointing {
  private var _x = 0
  private var _y = 0
  private val bound = 100

  def x = _x
  def x_= (newValue: Int): Unit = {
    if (newValue < bound) _x = newValue else printWarning
  }

  def y = _y
  def y_= (newValue: Int): Unit = {
    if (newValue < bound) _y = newValue else printWarning
  }
    
    private def printWarning = println("WARNING: Out of bounds")
  }
  
  def main(args : Array[String]) {
      val point1 = new Pointing
      point1.x = 99
      point1.y = 101
      println(point1.x)
      println(point1.y)
  }
}