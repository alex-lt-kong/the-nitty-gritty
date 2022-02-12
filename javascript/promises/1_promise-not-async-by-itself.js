
console.log('The line BEFORE the Promise() runs at: ', new Date(Date.now()).toISOString());
new Promise((resolve, reject) => {

  fs = require('fs');
  for (let i = 0; i < 100000; i ++) {
    // The loop is to simulate a long-running synchronous task
    fs.writeFileSync('/dev/null', i);
  }  
  console.log('writeFileSync() finishes at:           ', new Date(Date.now()).toISOString());
  resolve();
});


console.log('The line AFTER the Promise() runs at:  ', new Date(Date.now()).toISOString());

/*
> node 1_promise-not-async-by-itself.js 
The line BEFORE the Promise() runs at:  2022-02-12T09:45:47.710Z
writeFileSync() finishes at:            2022-02-12T09:45:58.242Z
The line AFTER the Promise() runs at:   2022-02-12T09:45:58.242Z

*/