var F = function () {
  this.a = 1;
  this.b = 2;
};
var prototypeOfF = Object.getPrototypeOf(F);
console.log(`The prototype of F is`, prototypeOfF);

var prototypeOfPrototypeOfF = Object.getPrototypeOf(prototypeOfF);
console.log(
  `The prototype of the prototype of F is`,
  prototypeOfPrototypeOfF, 
  '(i.e., Object.prototype, Object.prototype === prototypeOfPrototypeOfF is',
  Object.prototype === prototypeOfPrototypeOfF, ')');

  var prototypeOfPrototypeOfPrototypeOfF =Object.getPrototypeOf(prototypeOfPrototypeOfF);
console.log(
  `The prototype of the prototype of the prototype of F is`,
  prototypeOfPrototypeOfPrototypeOfF);