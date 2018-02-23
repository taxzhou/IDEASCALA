package basic

class TestAny {
  val list: List[Any] = List(
      "a String ",
      7111,
      'c',
      true,
      (x: Int) => x + 1
  )
  list.foreach(element => print(element))
}