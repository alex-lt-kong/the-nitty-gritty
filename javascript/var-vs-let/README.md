```
> node main.js 
functionScopeVsBlockScope
Hello World!
inside an enclosing block
Hello World!
foo bar
enclosing block ends
foo
Error caught:
 ReferenceError: d is not defined
    at functionScopeVsBlockScope (/mnt/hdd0/repos/the-nitty-gritty/javascript/var-vs-let/main.js:22:15)
    at Object.<anonymous> (/mnt/hdd0/repos/the-nitty-gritty/javascript/var-vs-let/main.js:82:3)
    at Module._compile (internal/modules/cjs/loader.js:999:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:1027:10)
    at Module.load (internal/modules/cjs/loader.js:863:32)
    at Function.Module._load (internal/modules/cjs/loader.js:708:14)
    at Function.executeUserEntryPoint [as runMain] (internal/modules/run_main.js:60:12)
    at internal/main/run_main_module.js:17:47

functionScopeCouldBeConfusing
My value: 3
My value: 3
My value: 3

blockScopeIsMoreIntuitive
My value: 0
My value: 1
My value: 2

varVariablesWillBeHoisted
undefined
hello world!

letVariablesWillAlsoBeHoistedLol
Error caught:
 ReferenceError: Cannot access 'a' before initialization
    at letVariablesWillAlsoBeHoistedLOL (/mnt/hdd0/repos/the-nitty-gritty/javascript/var-vs-let/main.js:69:15)
    at Object.<anonymous> (/mnt/hdd0/repos/the-nitty-gritty/javascript/var-vs-let/main.js:109:3)
    at Module._compile (internal/modules/cjs/loader.js:999:30)
    at Object.Module._extensions..js (internal/modules/cjs/loader.js:1027:10)
    at Module.load (internal/modules/cjs/loader.js:863:32)
    at Function.Module._load (internal/modules/cjs/loader.js:708:14)
    at Function.executeUserEntryPoint [as runMain] (internal/modules/run_main.js:60:12)
    at internal/main/run_main_module.js:17:47

```