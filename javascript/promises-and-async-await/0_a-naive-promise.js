/*
  Promises are one way to deal with asynchronous code, without getting stuck in
  callback hell, such as the following pseudo-code of pizza handling:

  chooseToppings(function(toppings) {
  placeOrder(toppings, function(order) {
    collectOrder(order, function(pizza) {
      eatPizza(pizza);
    }, failureCallback);
  }, failureCallback);
}, failureCallback);

*/

let done = true;

let isItDoneYet = new Promise((resolve, reject) => {
  /*
  The (anonymous) function passed to new Promise is called the executor.
  When new Promise is created, the executor runs automatically.
  It contains the producing code which should eventually produce the result.
  Its arguments resolve and reject are callbacks provided by JavaScript itself.
  Our code is only inside the executor.
  */
  if (done) {
    const workDone = 'Here is the thing I built'
    resolve(workDone)
  } else {
    const why = 'Still working on something else'
    reject(why)
  }
});


isItDoneYet
  .then(ok => {
    console.log(ok)
  })
  .catch(err => {
    console.error(err)
  });

/*
  Once a promise has been called, it will start in a pending state.
  This means that the calling function continues executing, while the
  promise is pending until it resolves, giving the calling function
  whatever data was being requested.
*/
