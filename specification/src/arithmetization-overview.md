# Arithmetization Overview

## Tables

<!-- auto-gen info spec_has_correct_table_overview -->
<!-- To reproduce this code, please run `cargo run spec_has_correct_table_overview`. -->
| table name                            | #main cols | #aux cols | total width |
|:--------------------------------------|-----------:|----------:|------------:|
| [ProgramTable](program-table.md)      |          7 |         3 |          16 |
| [ProcessorTable](processor-table.md)  |         39 |        11 |          72 |
| [OpStack](operational-stack-table.md) |          4 |         2 |          10 |
| [RamTable](random-access-memory-table.md) |          7 |         6 |          25 |
| [JumpStackTable](jump-stack-table.md) |          5 |         2 |          11 |
| [Hash](hash-table.md)                 |         67 |        20 |         127 |
| [Cascade](cascade-table.md)           |          6 |         2 |          12 |
| [LookupTable](lookup-table.md)        |          4 |         2 |          10 |
| [U32Table](u32-table.md)              |         10 |         1 |          13 |
| DegreeLowering                        |        220 |        36 |         328 |
| Randomizers                           |          0 |         1 |           3 |
| **TOTAL**                             |    **369** | **86**    |     **627** |
<!-- auto-gen info stop -->

## Overview

<!-- auto-gen info spec_has_correct_constraints_overview -->
<!-- To reproduce this code, please run `cargo run spec_has_correct_constraints_overview`. -->
| table name                                     | #initial | #consistency | #transition | #terminal | max degree |
|:-----------------------------------------------|---------:|-------------:|------------:|----------:|-----------:|
| [ProgramTable](program-table.md)               |        6 |            4 |          10 |         2 |          4 |
| [ProcessorTable](processor-table.md)           |       29 |           10 |          69 |         1 |         19 |
| [OpStack](operational-stack-table.md)          |        3 |            0 |           5 |         0 |          4 |
| [RamTable](random-access-memory-table.md)      |        7 |            0 |          12 |         1 |          5 |
| [JumpStackTable](jump-stack-table.md)          |        6 |            0 |           6 |         0 |          4 |
| [Hash](hash-table.md)                          |       22 |           45 |          47 |         2 |          9 |
| [Cascade](cascade-table.md)                    |        2 |            1 |           3 |         0 |          4 |
| [LookupTable](lookup-table.md)                 |        3 |            1 |           4 |         1 |          3 |
| [U32Table](u32-table.md)                       |        1 |           15 |          22 |         2 |         12 |
| [Grand Cross-Table Argument](table-linking.md) |        0 |            0 |           0 |        14 |          1 |
| **TOTAL**                                      |   **79** |       **76** |     **178** |    **23** |     **19** |
<!-- auto-gen info stop -->
