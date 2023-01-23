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
|    `swap_digest` |       |        |       |
|  `assert_vector` |       |        |       |
|    `absorb_init` |       |        |       |
|         `absorb` |       |        |       |
|        `squeeze` |       |        |       |
|            `add` |   ⨯   |        |       |
|            `mul` |   ⨯   |        |       |
|         `invert` |       |        |       |
|             `eq` |   ⨯   |        |       |
|          `split` |   ⨯   |        |       |
|             `lt` |   x   |        |       |
|            `and` |   x   |        |       |
|            `xor` |   x   |        |       |
|    `log_2_floor` |       |        |       |
|            `pow` |   x   |        |       |
|            `div` |       |        |       |
|          `xxadd` |       |        |       |
|          `xxmul` |       |        |       |
|        `xinvert` |       |        |       |
|          `xbmul` |   ⨯   |        |       |
|        `read_io` |   ⨯   |        |       |
|       `write_io` |   ⨯   |        |       |
