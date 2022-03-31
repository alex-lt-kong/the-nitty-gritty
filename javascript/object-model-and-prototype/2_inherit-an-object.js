const myObj = {
  myMember: 3,
  myFunc: function () {
    console.log('FooBar');
  }
};
console.log(
  'Is a and b myObj\'s "own property"? Object.getOwnPropertyNames(myObj):',
  Object.getOwnPropertyNames(myObj)
);

const myObk = Object.create(myObj); // myObk inherits from myObj

console.log(
  'Does myObk own myMember? See what does it own Object.getOwnPropertyNames(myObk):',
  Object.getOwnPropertyNames(myObk)
);
console.log('But still we can print myObk.myMember: ', myObk.myMember);
myObj.myMember = 4;
console.log(
  'Since myObk does not own myMember, the value changes as myObj.myMember changes:',
  myObk.myMember
);

console.log('We can even dynamically add a member to myObj');
myObj.myMembes = 'Hello world!';
console.log('it will still be accessible from myObk.myMembes:', myObk.myMembes);

myObk.myFunc();