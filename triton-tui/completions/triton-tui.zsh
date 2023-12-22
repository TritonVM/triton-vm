#compdef triton-tui

autoload -U is-at-least

_triton-tui() {
    typeset -A opt_args
    typeset -a _arguments_options
    local ret=1

    if is-at-least 5.2; then
        _arguments_options=(-s -S -C)
    else
        _arguments_options=(-s -C)
    fi

    local context curcontext="$curcontext" state line
    _arguments "${_arguments_options[@]}" \
'-i+[File containing public input]:PATH: ' \
'--input=[File containing public input]:PATH: ' \
'-n+[JSON file containing all non-determinism]:PATH: ' \
'--non-determinism=[JSON file containing all non-determinism]:PATH: ' \
'--initial-state=[JSON file containing entire initial state]:PATH: ' \
'--interrupt-cycle=[The maximum number of cycles to run after any interaction, preventing a frozen TUI in infinite loops]:u32: ' \
'-h[Print help]' \
'--help[Print help]' \
'-V[Print version]' \
'--version[Print version]' \
':program -- File containing the program to run:' \
&& ret=0
}

(( $+functions[_triton-tui_commands] )) ||
_triton-tui_commands() {
    local commands; commands=()
    _describe -t commands 'triton-tui commands' commands "$@"
}

if [ "$funcstack[1]" = "_triton-tui" ]; then
    _triton-tui "$@"
else
    compdef _triton-tui triton-tui
fi
