# Arithmetization Overview

## Tables

<!-- auto-gen info spec_has_correct_table_overview -->
<!-- To reproduce this code, please run `cargo run spec_has_correct_table_overview`. -->
| table name                                 | #main cols | #aux cols | total width |
|:-------------------------------------------|-----------:|----------:|------------:|
| [ProgramTable](program-table.md)           |          7 |         3 |          16 |
| [ProcessorTable](processor-table.md)       |         39 |        11 |          72 |
| [OpStackTable](operational-stack-table.md) |          4 |         2 |          10 |
| [RamTable](random-access-memory-table.md)  |          7 |         6 |          25 |
| [JumpStackTable](jump-stack-table.md)      |          5 |         2 |          11 |
| [HashTable](hash-table.md)                 |         67 |        20 |         127 |
| [CascadeTable](cascade-table.md)           |          6 |         2 |          12 |
| [LookupTable](lookup-table.md)             |          4 |         2 |          10 |
| [U32Table](u32-table.md)                   |         10 |         1 |          13 |
| DegreeLowering                             |        217 |        36 |         325 |
| Randomizers                                |          0 |         1 |           3 |
| **TOTAL**                                  |    **366** |    **86** |     **624** |
<!-- auto-gen info stop -->

## Constraints

The following table captures the state of affairs in terms of constraints before automatic degree lowering.
In particular, automatic degree lowering introduces new columns, modifies the constraint set (in a way that
is equivalent to what was there before), and lowers the constraints' maximal degree.

<!-- auto-gen info spec_has_correct_constraints_overview -->
<!-- To reproduce this code, please run `cargo run spec_has_correct_constraints_overview`. -->

Before automatic degree lowering:

| table name                                     | #initial | #consistency | #transition | #terminal | max degree |
|:-----------------------------------------------|---------:|-------------:|------------:|----------:|-----------:|
| [ProgramTable](program-table.md)               |        6 |            4 |          10 |         2 |          4 |
| [ProcessorTable](processor-table.md)           |       29 |           10 |          69 |         1 |         19 |
| [OpStackTable](operational-stack-table.md)     |        3 |            0 |           5 |         0 |          4 |
| [RamTable](random-access-memory-table.md)      |        7 |            0 |          12 |         1 |          5 |
| [JumpStackTable](jump-stack-table.md)          |        6 |            0 |           6 |         0 |          4 |
| [HashTable](hash-table.md)                     |       22 |           45 |          47 |         2 |          9 |
| [CascadeTable](cascade-table.md)               |        2 |            1 |           3 |         0 |          4 |
| [LookupTable](lookup-table.md)                 |        3 |            1 |           4 |         1 |          3 |
| [U32Table](u32-table.md)                       |        1 |           15 |          22 |         2 |         12 |
| [Grand Cross-Table Argument](table-linking.md) |        0 |            0 |           0 |        14 |          1 |
| **TOTAL**                                      |   **79** |       **76** |     **178** |    **23** |     **19** |

After automatically lowering degree to 4:

| table name                                     | #initial | #consistency | #transition | #terminal |
|:-----------------------------------------------|---------:|-------------:|------------:|----------:|
| [ProgramTable](program-table.md)               |        6 |            4 |          10 |         2 |
| [ProcessorTable](processor-table.md)           |       31 |           10 |         252 |         1 |
| [OpStackTable](operational-stack-table.md)     |        3 |            0 |           5 |         0 |
| [RamTable](random-access-memory-table.md)      |        7 |            0 |          13 |         1 |
| [JumpStackTable](jump-stack-table.md)          |        6 |            0 |           6 |         0 |
| [HashTable](hash-table.md)                     |       22 |           52 |          84 |         2 |
| [CascadeTable](cascade-table.md)               |        2 |            1 |           3 |         0 |
| [LookupTable](lookup-table.md)                 |        3 |            1 |           4 |         1 |
| [U32Table](u32-table.md)                       |        1 |           26 |          34 |         2 |
| [Grand Cross-Table Argument](table-linking.md) |        0 |            0 |           0 |        14 |
| **TOTAL**                                      |  **158** |      **152** |     **356** |    **46** |
<!-- auto-gen info stop -->
