<div align="center">
<img src="./assets/icon.svg" width="100" height="100" />
<h1>plasma</h1>
</div>

Idk i was kinda having problems with writing circuits again and again for torch. Note that this is not
the rewriting of anything anywhere and is merely a wrapper. I am wholely bound by Qiskit for
the actual bottleneck in speed.

**This for better DX and not more speed**

## I think i know what happened the last time
- Using bitstrings is a bad idea: 0010000 -> 0001000 is not a 2 bit change, its a 24 change (16+8).