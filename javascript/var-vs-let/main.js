/* The main difference is scoping rules. Variables declared by var keyword are
   scoped to the immediate function body (hence the function scope) while let 
   variables are scoped to the immediate enclosing block denoted by { } (hence
   the block scope). */

function functionScopeVsBlockScope() {
  var a = "Hello";
  let b = "World!";

  console.log(a, b);

  {
    var c = "foo"
    let d = "bar";
    console.log('inside an enclosing block');
    console.log(a, b);
    console.log(c, d);
  }
  console.log('enclosing block ends');

  console.log(c); // still okay
  console.log(d); // ReferenceError: d is not defined
}

function functionScopeCouldBeConfusing() {
  var funcs = [];
  for (var i = 0; i < 3; i++) {
    funcs[i] = function() {
      console.log("My value: " + i);
    };
  }
  for (var j = 0; j < 3; j++) {
    funcs[j]();
  }
  /* it prints:
     My value: 3
     My value: 3
     My value: 3
     instead of:
     My value: 0
     My value: 1
     My value: 2
     the problem is that the variable i, within each of your anonymous
     functions, is bound to the same variable outside of the function.
  */
}

function blockScopeIsMoreIntuitive() {
  var funcs = [];
  for (let i = 0; i < 3; i++) {
    // with a let-based index, each iteration through the loop will have a new
    // variable i with loop scope, so your code would work as you expect.
    funcs[i] = function() {
      console.log("My value: " + i);
    };
  }
  for (var j = 0; j < 3; j++) {
    funcs[j]();
  }
}

function varVariablesWillBeHoisted() {
  console.log(a); // It prints undefined instead of throws an exception
  var a = 'hello world!';
  console.log(a);
}

function letVariablesWillAlsoBeHoistedLOL() {
  console.log(a);
  /*
    Variables declared with let and const are also hoisted but, unlike var,
    are not initialized with a default value. An exception will be thrown 
    if a variable declared with let or const is read before it is initialized. 
  */
  let a = 'hello world!';
  console.log(a);
}

try {
  func = functionScopeVsBlockScope
  console.log(func.name);
  func();
}
catch (error) {
  console.error('Error caught:\n', error);
}
finally {
  console.log();
}

func = functionScopeCouldBeConfusing
console.log(func.name);
func();
console.log();

func = blockScopeIsMoreIntuitive
console.log(func.name);
func();
console.log();

func = varVariablesWillBeHoisted
console.log(func.name);
func();
console.log();

try {
  func = letVariablesWillAlsoBeHoistedLol;
  console.log(func.name);
  func();
}
catch (error) {
  console.error('Error caught:\n', error);
}
finally {
  console.log();
}