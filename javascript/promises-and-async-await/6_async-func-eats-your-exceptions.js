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


console.log('BEFORE async doubleAdd()');
async function doubleAdd(a, b, c) {
  let sum0 = await add(a, b);
  console.log(sum0);
  let sum1 = await add(sum0, c);
  console.log(sum1);
  // In the previous example, these lines are enclosed in a try...catch block
  // but here they aren't. In the past, the exception here will be silently
  // swallowed. But seems JavaScript evolved a bit since then and now we
  // get a message for that.
}
doubleAdd(1, 2, -1);
console.log('AFTER async doubleAdd()');