# Reflection

* No, C++ doesn't have reflection support.

* But what is reflection anyway?
    * Theoretically, it means "the ability of a process to examine, introspect, and
    modify its own structure and behavior".
    * Practically it is very straightforward--a class can print and change its member methods and variables in runtime.
    Below is an example in JavaScript.
    ```JavaScript
    var person = {
    fname: "Default",
    lname: "Default",
        getFullName: function(){
            return this.fname + " " + this.lname;
        }
    }
    var john = {
        fname: "John",
        lname: "Doe"
    }
    john.__proto__ = person;

    for(var prop in john){
        console.log(prop + " : " + john[prop]);
    }

    >>> fname : John
    >>> lname : Doe
    >>> getFullName : function(){
    >>>     return this.fname + " " + this.lname;
    >>> }
    ```

* C++ does not have it, why? Answers from [this post](https://stackoverflow.com/questions/359237/why-does-c-not-have-reflection)
provide some clues:
  * It's a lot of work to add, and the C++ committee is fairly conservative, and doesn't spend time on radical
  new features unless they're sure it'll pay off.

  * C++ makes very few guarantees about the compiled code. The compiler is allowed to do pretty much anything
  it likes, as long as the resulting functionality is what is expected. For example, your classes aren't
  required to actually be there. The compiler can optimize them away, inline everything they do, and it
  frequently does just that, because even simple template code tends to create quite a few template instantiations.
  The C++ standard library relies on this aggressive optimization. operator `[]` on a vector is only comparable to
  raw array indexing in performance because the entire operator can be inlined and thus removed entirely from the
  compiled code.
    * In comparison, C# and Java make a lot of guarantees about the output of the compiler. If I define a class
    in C#, then that class will exist in the resulting assembly. Even if I never use it. Even if all calls to
    its member functions could be inlined. The class has to be there, so that reflection can find it. If you were
    allowed to inspect the metadata of a C++ executable, you'd expect to see every class it defined, which
    means that the compiler would have to preserve all the defined classes, even if they're not necessary.

  * C++ (and C for sure) is essentially a very sophisticated macro assembler--in C/C++, programmers are easily
  aware of the strong correlation between their code and the underlying machine's operation. In other words, what
  things can be done and the way they are done in C/C++ depends heavily on what things can be done and the way
  they are done in machine code. C/C++'s contribution is to extend Assembler technology to incredible
  levels of complexity management, and abstractions to make programming larger, more complex tasks
  vastly more possible for human beings.
    * In this sense, C++ is **NOT** (in a traditional sense) a high-level language like C#, Java, Objective-C,
    Smalltalk, etc. C#, Java, Objective-C all require a much larger, richer runtime system to support
    their execution.

  * Reflection for languages that have it is about how much of the source code the compiler is willing to
  leave in your object code to enable reflection. Unless the compiler keeps all the source code around,
  reflection will be limited. The C++ compiler doesn't keep anything around (well, ignoring RTTI),
  so you don't get reflection at the moment.
    * In the case of Java and C# compilers only keep class, method names and return types
    around, so you get a little bit of reflection data, but you can't inspect expressions or program structure.
    * In the case of JavaScript/Python, as they are interpreted languages, interpreters are happy to provide
    full-fledged reflection functionality--as source code is already there.
