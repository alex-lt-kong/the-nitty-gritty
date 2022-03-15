class NaiveDict {

  constructor(enableAsync) {
    this.keys = new Array(10);
    this.values = new Array(10);
    this.counter = 0;
    this.enableAsync = enableAsync;
  }
  
  async append(key) {
    for (let i = 0; i < this.keys.length; i++) {
      if (this.keys[i] == key) {
        this.values[i] = this.values[i] + 1;
        return;
      }
    }
    if (this.counter < this.keys.length) {
      if (this.enableAsync) { await new Promise(r => setTimeout(r, 0)); }
      this.keys[this.counter] = key;
      this.values[this.counter] = 1;
      this.counter += 1;
    }
  }

  remove(key) {
    for (let i = 0; i < this.keys.length; i++) {
      if (this.keys[i] == key) {
        this.values[i] = this.values[i] - 1;
      }
    }
  }

  show() {
    console.log(this.keys, this.values);
  }
}

var myArray = ['Jan', 'Feb', 'Mar', 'Apr', 'May'];
var myDict = new NaiveDict(false);

async function longOperation() {

  const randomElement = myArray[Math.floor(Math.random() * myArray.length)];
  myDict.append(randomElement);  
  myDict.show();
  await new Promise(r => setTimeout(r, Math.random()));
  myDict.remove(randomElement); 
  myDict.show();
}

for (let i = 0; i < 10; i ++) {
  longOperation();
}




