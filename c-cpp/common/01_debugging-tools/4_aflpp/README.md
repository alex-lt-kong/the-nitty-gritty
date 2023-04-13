# AFL++
* Fuzzing or fuzz testing is an automated software testing technique
that involves providing invalid, unexpected, or random data as inputs
to a computer program.

* Install `apt install afl++`
    * `apt install gnuplot` for `afl-plot`

* Check:
    * current test case: `cat ./output/.cur_input`
    * crashes: `cat ./output/crashes/`
    * plots: `afl-plot ./output/ ./html`