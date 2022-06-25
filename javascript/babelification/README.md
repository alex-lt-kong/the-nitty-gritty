To Babelify the source code `index.src.js` to `index.js` use the following
js script according to the link https://github.com/babel/babelify.
```
var fs = require("fs");
var browserify = require("browserify");
browserify("./index.src.js")
  .transform("babelify", {presets: ["@babel/preset-env", "@babel/preset-react"]})
  .bundle()
  .pipe(fs.createWriteStream("index.js"));
```