/*----------EEPROM------------*/
     #include <EEPROM.h>
   void writeEEPROM(String ch,int from)
   {  
    for (int i=0;ch[i];i++)
    {
      EEPROM.write(from+i,int(ch[i]));
    }
   }
   String readEEPROM(int from,int to )
   {
      String ch ="";
      for (int i=from;i<to+1;i++)
      {
        int ascii = EEPROM.read(i);
        char c = char(ascii);
        ch+=c;
        
      }
      return ch;
   }
  String engineId = "-1";
  String maintenanceIndex ="-1";
  String cycle ="-1";
  String lastCycleReached ="-1";


/*----MMA7455 Accelerometer--------*/
  // Including Libraries
      #include <Wire.h> //Include the Wire library
      #include <MMA_7455.h> //Include the MMA_7455 library
  // Sensor Object Declaration 
      MMA_7455 accelerometer = MMA_7455();

/*----DHT22 Temperature/Humidity--------*/
  // Including Libraries
        #include "DHT.h"
  // Pin Definition
      #define dhtPin 8 // broche ou l'on a branche le capteur
  // Dht Sensor Type definition : DHT 22 (AM2302)
      #define dhtType DHT22 
   // Sensor Object Declaration
      DHT dht(dhtPin, dhtType);//déclaration du capteur
/*--------- BlueTooth HC-05-----------*/
  // Including Libraries
      #include <SoftwareSerial.h>
  // Pin Definition
      #define rxPin 2
      #define txPin 3
  // HC-05 Sensor Object declaration
      SoftwareSerial hc05(rxPin ,txPin);
  // bluetooth frequency 
      #define baudrate 9600
/*--------------------------------*/
String message , response;
/*-------measurement--------------*/
char xVal, yVal, zVal; // MMA7455 Accelerometer
float hygrometry,temperatureC , temperatureF , temperatureIndexC ,temperatureIndexF; // DHT22
/*-------Setting----------------*/
#define frequency 9600 // 9600 Hz
#define wait 2000 // 1000ms = 1s
/*--------------------------------------*/
void measure()
{
   // MMA7455 Accelerometer 
  xVal = accelerometer.readAxis('x'); //Read the ‘x’ Axis
  yVal = accelerometer.readAxis('y'); //Read the ‘y’ Axis
  zVal = accelerometer.readAxis('z'); //Read the ‘z’ Axis
  // DHT 22
 hygrometry = dht.readHumidity();//hygrometry 
 temperatureC = dht.readTemperature();//temperature C°
 temperatureF = dht.readTemperature(true);//temperature F°
 temperatureIndexF = dht.computeHeatIndex(temperatureF, hygrometry);//  temperatureIndex F°
 temperatureIndexC = dht.computeHeatIndex(temperatureC, hygrometry, false); // temperatureIndex C°
}
void printMesurement()
{
 // MMA7455 Accelerometer 
  Serial.print("X = ");
  Serial.print(xVal, DEC);
  Serial.print(" Y = ");
  Serial.print(yVal, DEC);
  Serial.print(" Z = ");
  Serial.println(zVal, DEC);
  // DHT 22
 Serial.print("Humidite: ");
 Serial.print(hygrometry);
 Serial.print(" %\t");
 Serial.print("Temperature: ");
 Serial.print(temperatureC);
 Serial.print(" *C ");
 Serial.print(temperatureF);
 Serial.print(" *F\t");
 Serial.print("Indice de temperature: ");
 Serial.print(temperatureIndexC);
 Serial.print(" *C ");
 Serial.print(temperatureIndexF);
 Serial.println(" *F");
}

bool isWrong()
{
  if (isnan( hygrometry) || isnan(temperatureC) || isnan(temperatureIndexF))
 {
   Serial.println("Failed to read from DHT sensor!");
   return true;
 }
 if (isnan(xVal) || isnan(yVal) || isnan(zVal))
 {
   Serial.println("Failed to read from Accelerometer sensor!");
   return true;
 }
 return false;
}
String convertFloatToString(float temperature)
{ // begin function

  char temp[10];
  String tempAsString;
    
    // perform conversion
    dtostrf(temperature,1,2,temp);
    
    // create string object
  tempAsString = String(temp);
  
  return tempAsString;
  
} // end function
 void writeString(String stringData) { // Used to serially push out a String with Serial.write()

  for (int i = 0; i < stringData.length(); i++)
  {
    Serial.write(stringData[i]);   // Push each char 1 by 1 on each loop pass
  }
  }
void sendMsg()
{
 
// message = ["ax,ay,az,]
  message = convertFloatToString(xVal)+","+convertFloatToString(yVal)+","+convertFloatToString(zVal)+","+convertFloatToString(temperatureC)+","+convertFloatToString(hygrometry)+","+convertFloatToString(temperatureIndexC);
  hc05.println(message);
  }
void receiveRsp()
{
  response = "";
  while (hc05.available()>0){
     response += hc05.readString();
    // Serial.write(hc05.read());
     
    }
    if (response =="") return;
    writeString("serveur response => ");
    writeString(response);
    Serial.write('\n');
}

/*----------------------*/
void setup() {

pinMode(rxPin,INPUT);
pinMode(txPin,OUTPUT);

Serial.begin(frequency);
delay(500);

Serial.println("init MMA7455.");
accelerometer.initSensitivity(2);
delay(500);

Serial.println("init DHT22 .");
 dht.begin();
  delay(500);

Serial.println("init HC-05 .");
  hc05.begin(baudrate);



}
void loop() {
 
 /*-------------------------------*/
measure();
if (isWrong()) return;
//printMesurement();
receiveRsp();
sendMsg();
delay(wait);
  
}
