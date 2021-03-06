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

console.log('BEFORE Promise()');
add(1, 2).then((sum) => {     
  console.log(sum);
  return add(sum, 4);
}).then(
  sum => console.log(sum)
).catch((e) => { 
  console.log(e) 
});
console.log('AFTER Promise()');

console.log('BEFORE async doubleAdd()');
async function doubleAdd(a, b, c) {
  try {
    let sum0 = await add(a, b);
    console.log(sum0);
    let sum1 = await add(sum0, c);
    console.log(sum1);

  } catch (error) {
    console.log(error);
  }
}
doubleAdd(1, 2, 4);
console.log('AFTER async doubleAdd()');