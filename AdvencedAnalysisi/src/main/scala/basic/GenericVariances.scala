package basic

object GenericVariances {
  class Stack[A] {
    private var elements: List[A] = Nil
    def push(x:A) {elements = x :: elements }
    def peek: A = elements.head
    def pop(): A = {
      val currentTop = peek
      elements = elements.tail
      currentTop
    }
  }
  
  class Fruit 
  class Apple extends Fruit
  class Banana extends Fruit
  
  abstract class Animal {
    def name: String
  }
  
  case class Cat(name:String) extends Animal
  case class Dog(name:String) extends Animal
  def printAnimalNames(animals: List[Animal]): Unit = {
    animals.foreach {
       animal => println(animal.name)
    }
  }
  val cats: List[Cat] = List(Cat("Whiskers"),Cat("Tom"))
  val dogs: List[Dog] = List(Dog("Fido"),Dog("Rex"))
  
  
  abstract class Printer[-A] {
    def print(value: A):Unit
  }
  class AnimalPrinter extends Printer[Animal] {
    def print(animal:Animal):Unit = 
      println("The animal's name is :" + animal.name)
  }
  class CatPrinter extends Printer[Cat] {
    def print(cat:Cat):Unit = 
      println("The cat's name is :" + cat.name)
  }
  val myCat : Cat = Cat("Boots")
  
  val catPrinter: Printer[Cat] = new CatPrinter
  val animalPrinter: Printer[Animal] = new AnimalPrinter
  
  def printMyCat(printer:Printer[Cat]):Unit={
    printer.print(myCat)
  }
  
  
  /**
   * Inner Classes
   * */
  class Graph{
    class Node {
      var connectedNodes: List[Graph#Node] = Nil
      def connectTo(node:Graph#Node) {
        if (connectedNodes.find(node.equals).isEmpty) {
          connectedNodes = node :: connectedNodes
        }
      }
    }
    var nodes: List[Node] =Nil
    def newNode : Node = {
      val res = new Node
      nodes = res::nodes
      res
    }
  }
  
  def main(args: Array[String]): Unit = {
    val stack = new Stack[Int]
    stack.push(1)
    stack.push(2)
    println(stack.pop)
    println(stack.pop)
    
    val fruitStack = new Stack[Fruit]
    val app = new Apple
    val banana = new Banana
    fruitStack.push(app)
    fruitStack.push(banana)
    println(fruitStack.pop())
    println(fruitStack.pop())
    
    
    printAnimalNames(cats)
    printAnimalNames(dogs)
    
    printMyCat(catPrinter)
    printMyCat(animalPrinter)
    
    println("____________________")
    
    val graph1: Graph = new Graph
    val node1:graph1.Node = graph1.newNode
    val node2:graph1.Node = graph1.newNode
    val node3:graph1.Node = graph1.newNode
    node1.connectTo(node2)
    node1.connectTo(node3)
    println(graph1.nodes)
    println(node1.connectedNodes)
    
    val graph2 : Graph = new Graph
    val node4:graph2.Node = graph2.newNode
    node1.connectTo(node4)
    println(node4)
    println(node1.connectedNodes)    
    
  }
  
  
}