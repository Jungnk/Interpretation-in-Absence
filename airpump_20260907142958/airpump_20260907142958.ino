int pumpRelay = 7;   // pump A 
int pumpRelay2 = 8;  // pump B 
int pumpRelay3 = 9;  // pump C 

const unsigned long inflateTime = 10000;  //  (ms)
const unsigned long deflateTime = 10000;  //  (ms)

void setup() {
  pinMode(pumpRelay, OUTPUT);
  pinMode(pumpRelay2, OUTPUT);
  pinMode(pumpRelay3, OUTPUT);

  digitalWrite(pumpRelay, LOW);
  digitalWrite(pumpRelay2, HIGH); // pin8: 
  digitalWrite(pumpRelay3, HIGH); // pin9: 
}

void loop() {
  digitalWrite(pumpRelay, HIGH);   // pin7 ON
  delay(inflateTime);

  digitalWrite(pumpRelay, LOW);    // pin7 OFF
  delay(deflateTime);

}