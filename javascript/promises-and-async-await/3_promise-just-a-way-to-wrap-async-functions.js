
console.log('The line BEFORE the Promise() runs at: ', new Date(Date.now()).toISOString());
var myPromise = new Promise((resolve, reject) => {

  setTimeout(
    function() {
      resolve(new Date(Date.now()).toISOString());
    }, 
  10000);
  
  ;
});

myPromise
  .then(msg => {
    console.log('writeFileSync() finishes at:           ', msg);
  })
  .catch(err => {
    console.log(err);
  });
// .then()/.catch() blocks in promises are basically the async equivalent of
// a try...catch block in sync code. Bear in mind that synchronous try...catch
// won't work in async code.

// However, while asynchronous functions are everywhere in Javascript, it is,
// just like Python, single-threaded.

console.log('The line AFTER the Promise() runs at:  ', new Date(Date.now()).toISOString());

/*
> node 3_promise-just-a-way-to-wrap-async-functions.js 
The line BEFORE the Promise() runs at:  2022-02-12T09:44:31.036Z
The line AFTER the Promise() runs at:   2022-02-12T09:44:31.061Z
writeFileSync() finishes at:            2022-02-12T09:44:41.061Z

*/