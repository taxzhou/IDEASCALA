package basic

import scala.util.Random

object Casesclasses {
  case class Book(isbn: String)
  val frankenstein = Book("987-0486282114")
  
  val ainstein = Book("987-0486282114")
  
  val kena = Book(ainstein.isbn)
  
  val x : Int = Random.nextInt(10)
  
  def matchTest(x : Int): String = x match {
    case 0 => "zero"
    case 1 => "one"
    case 2 => "two"
    case _ => "many"
  }
  
  abstract class Notification
  
  case class Email(sender:String,title:String,body:String) extends Notification
  
  case class SMS(caller:String,message:String) extends Notification
  
  case class VoiceRecording(contactName:String,link:String) extends Notification
  
  def showNotification(notification:Notification):String ={
    notification  match {
      case Email(email,title,_) =>
        s"You got an Email from $email with title $title"
      case SMS(number,message) =>
        s"You got an SMS from $number! Message:$message"
      case VoiceRecording(name,link) =>
        s"You've received a Voice Recording from $name! Click the link to hear it : $link"
    }
  }
  
  
  def showImportantNotification(notification:Notification,improtantPeopel:Seq[String]):String = {
    notification match {
      case Email(email,_,_) if improtantPeopel.contains(email) => 
        "You got an Email from Important people"
      case SMS(number,_) if improtantPeopel.contains(number) =>
        "you got an SMS from special someone"
      case other =>
        showNotification(other)
    }
  }
  
  
  abstract class Device 
  case class Phone(model:String) extends Device{
    def screenOff: String = "Turing screen off"
  }
  
  case class Computer(model:String) extends Device{
    def screenSaverOn = "Turing screen saver on..."
  }
  def goIdle(device:Device)  = device match {
    case p: Phone => p.screenOff
    case c: Computer => c.screenSaverOn
  }
  
  sealed abstract class Furniture 
  case class Couch() extends Furniture
  case class Chair() extends Furniture

  def findPlaceSit(piece : Furniture):String = piece match {
    case a:Couch => "Lie on the Couch"
    case b:Chair => "Lie on the Chair"
  }
  
  
  def main(args: Array[String]): Unit = {
    println(frankenstein.isbn)
    println(frankenstein == ainstein)
    println(kena.isbn)
    println(kena)
    println(matchTest(x))
    
    val somSMS = SMS("12345","Are you Ready")
    val someVoiceRecording = VoiceRecording("Tom","voiceRecording.org/id/123")
    val importantPeopleInfo = Seq("867-5309", "jenny@gmail.com")

    val importantEmail = Email("jenny@gmail.com", "Drinks tonight?", "I'm free after 5!")
    val importantSms = SMS("867-5309", "I'm here! Where are you?")

    println(showImportantNotification(somSMS, importantPeopleInfo))
    println(showImportantNotification(someVoiceRecording, importantPeopleInfo))
    println(showImportantNotification(importantEmail, importantPeopleInfo))
    println(showImportantNotification(importantSms, importantPeopleInfo))
    
    println("_________________________________________")
    
    
    println(goIdle(Computer("Computers")))
    
    
    println(findPlaceSit(Couch()))
    
    println("____Singleton Object paramaters_____")
    println(Casesclasses.ainstein)
  }
}