package tools

import org.apache.log4j.{Level, Logger}

object loggerLevel {

  /**
    * 设置默认的Logger级别，将spark中Info日志清楚
    * 将tomcat的日志关闭不显示
    */
  def setDefaultLoggerLevel(): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
  }

}
