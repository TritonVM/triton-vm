# Modified Memory Pointers by Instruction

|                  | `osp` | `ramp` | `jsp` |
|-----------------:|:-----:|:------:|:-----:|
|            `pop` |   ⨯   |        |       |
|     `push` + `a` |   ⨯   |        |       |
|         `divine` |   ⨯   |        |       |
|      `dup` + `i` |   ⨯   |        |       |
|     `swap` + `i` |       |        |       |
|            `nop` |       |        |       |
|           `skiz` |   ⨯   |        |       |
|     `call` + `d` |       |        |   ⨯   |
|         `return` |       |        |   ⨯   |
|        `recurse` |       |        |       |
|         `assert` |   ⨯   |        |       |
|           `halt` |       |        |       |
|       `read_mem` |       |   ⨯    |       |
|      `write_mem` |       |   ⨯    |       |
|           `hash` |       |        |       |
| `divine_sibling` |       |        |       |
|  `assert_vector` |       |        |       |
|            `add` |   ⨯   |        |       |
|            `mul` |   ⨯   |        |       |
|         `invert` |       |        |       |
|          `split` |   ⨯   |        |       |
|             `eq` |   ⨯   |        |       |
|            `lsb` |   ⨯   |        |       |
|          `xxadd` |       |        |       |
|          `xxmul` |       |        |       |
|        `xinvert` |       |        |       |
|          `xbmul` |   ⨯   |        |       |
|        `read_io` |   ⨯   |        |       |
|       `write_io` |   ⨯   |        |       |
