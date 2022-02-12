/*
  setTimeout() is an asynchronous function, meaning that the timer function will
  not pause execution of other functions in the functions stack.
  In other words, you cannot use setTimeout() to create a "pause" before the
  next function in the function stack fires. 
*/
setTimeout(
  function() {
    console.log(
      'The function INside a setTimeout() runs at:', 
      new Date(Date.now()).toISOString());
  }, 
3000);

console.log('The line AFTER setTimeout() runs at:       ', new Date(Date.now()).toISOString());

/*
> node 2_settimeout-is-async.js 
The line AFTER setTimeout() runs at:        2022-02-12T09:45:21.247Z
The function INside a setTimeout() runs at: 2022-02-12T09:45:24.249Z
*/