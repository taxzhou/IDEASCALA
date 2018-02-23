package basic

object CompoundSelfType {
  trait Cloneable extends java.lang.Cloneable {
    override def clone(): Cloneable = {
      super.clone().asInstanceOf[Cloneable]
    }
  }
  trait Resetable {
    def reset : Unit
  }
  def cloneAndReset(obj: Cloneable with Resetable): Cloneable = {
    val cloned = obj.clone()
    obj.reset
    cloned
  }
  
  /**
   * Start the self-Type
   */
  trait User {
    def username:String
  }
  
  trait Tweeter {
    ab: User =>
    def tweet(tweetText: String) = println(s"$username: $tweetText")
  }
  
  class VerifiedTweeter(val username_ : String) extends Tweeter with User {
    def username = s"real $username_"
  }
  
  
  def main(args: Array[String]): Unit = {
    val realBeyonce = new VerifiedTweeter("Beyonce")
    realBeyonce.tweet("Just spilled my glass of lemonade")
  }
}