package basic

object UpperLowerTypeBounds {
  abstract class Animal {
    def name: String
  }
  class Pet extends Animal {
    override def name:String = "Pet"
  }
  class Cat extends Pet {
    override def name: String = "Cat"
  }
  class Dog extends Pet{
    override def name = "Dog"
  }
  class Lion extends Animal {
    override def name = "Lion"
  }
  class PetContainer[P <: Pet](p:P) {
    def pet:P = p
  }
  
  
  trait Node[+B] {
    def prepend[U>:B](elem:U):Node[U]
  }
  case class ListNode[+B](h:B,t:Node[B]) extends Node[B] {
    def prepend[U>:B](elem:U) =ListNode[U](elem,this)
    def head:B = h
    def tail = t
  }
  case class Nil[+B]() extends Node[B] {
    def prepend[U>:B](elem: U) = ListNode[U](elem,this)
  }
  trait Bird
  case class AfricanSwallow() extends Bird
  case class EuropeanSwallow() extends Bird

  
  def main(args: Array[String]): Unit = {
    val dogContainer = new PetContainer[Dog](new Dog)
    val catContainer = new PetContainer[Cat](new Cat)
    
    val petContainer = new PetContainer[Pet](new Pet)
    
    println(petContainer.pet.name)
    
    //val lionContainer = new PetContainer[Lion](new Lion)
    
    
    println("\"End of the Upper Type Bounds\"")
    
    val africanSwallowList = ListNode[AfricanSwallow](AfricanSwallow(),Nil())
    var birdList: ListNode[Bird] = africanSwallowList
    var a = birdList.prepend(new EuropeanSwallow).prepend(new AfricanSwallow)
    
    println(birdList)
    println(a)
 }
}