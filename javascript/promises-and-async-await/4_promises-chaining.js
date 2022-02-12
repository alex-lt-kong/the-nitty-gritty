/*
  In the old days, doing several asynchronous operations in a row would lead
  to the classic callback pyramid of doom:

doSomething(function(result) {
  doSomethingElse(result, function(newResult) {
    doThirdThing(newResult, function(finalResult) {
      console.log('Got the final result: ' + finalResult);
    }, failureCallback);
  }, failureCallback);
}, failureCallback);

With modern functions, we attach our callbacks to the returned promises instead,
forming a promise chain:

doSomething()
.then(function(result) {
  return doSomethingElse(result);
})
.then(function(newResult) {
  return doThirdThing(newResult);
})
.then(function(finalResult) {
  console.log('Got the final result: ' + finalResult);
})
.catch(failureCallback);
*/

function add(a, b) {     
  return new Promise((resolve, reject) => {        
      setTimeout(() => {            
          if (a < 0 || b < 0) {                 
              return reject('Numbers must be non-negative')
          } 
          resolve(a + b)         
      }, 2000) 
  })
}

add(1, 2).then((sum) => {     
  console.log(sum);  // Print 3   
  return add(sum, 4);
  // note here we return another Promise() by calling add() and pass the result
  // from the previous Promise() as a parameter.
}).then((sum2) => {     
  console.log(sum2) // Print 7 
  return add(sum2, -1);
}).then((sum3) => {     
  console.log(sum3)
  // won't run since -1 triggers reject() instead of resolve();
}).catch((e) => { 
  console.log(e) 
});

/*
> node 4_promises-chaining.js 
3
7
Numbers must be non-negative
*/