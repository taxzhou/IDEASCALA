package basic

import scala.collection.mutable.ArrayBuffer

object UsingTraits {
  trait HairColor
  
  trait Iterator[A] {
    def hasNext: Boolean
    def next(): A
  }
  
  class IntIterartor(to: Int) extends Iterator[Int] {
    private var current = 7
    override def hasNext: Boolean = current < to
    override def next(): Int = {
      if (hasNext){
        val t = current 
        current +=1
        t
      } else 0
    }
  }
  
  trait Pet {
    val name : String 
  }
  class Cat(val name: String) extends Pet
  class Dog(val name: String) extends Pet
  
  val dog = new Dog("Dogs")
  val cat = new Cat("Cats")
  
  
  
  
  
  def main(args: Array[String]): Unit = {
    val iterator = new IntIterartor(10)
    println(iterator.next())
    println(iterator.next())
    
    
    val animals = ArrayBuffer.empty[Pet]
    animals.append(dog)
    animals.append(cat)
    animals.foreach(pet => print(pet.name + "   "))
  }
}