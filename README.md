# gpgpu-sim-exercise
Programming Assignment for ECE786. 

### Problem Description(s)
 
- Task 1:

This task requires you to modify the functional simulator. You need to change
the semantics of float(32-bits) point add instruction to power operation.
Suppose you have a float point add instruction fd = fa + fb. The register fa and
fb store 32-bits float point value. fa =2.0, fb=3.0, the original add instruction
will generate the new value 5.0 and store it to fd. Now the new semantic should
generate the value 8.0=2.0^3.0 and store it to fd.

- There is a function called `add_impl` which seems to be doing all sorts of
  addition, not just int or float type. Should we change the addition semantic
  only for float add or all different types?
