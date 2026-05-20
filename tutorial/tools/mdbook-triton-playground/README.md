# mdbook-triton-playground

A tiny mdBook preprocessor that rewrites `<triton-playground>...</triton-playground>` HTML islands so their contents are
wrapped in `<pre><code>...</code></pre>` before the book is rendered.

## Usage

Add this to `book.toml`:

```toml
[preprocessor.triton-playground]
command = "cargo run --quiet --manifest-path tools/mdbook-triton-playground/Cargo.toml --"
```

If your `book.toml` lives in a subdirectory, adjust the relative `manifest-path` accordingly.

## Notes

- The preprocessor only rewrites raw HTML islands in chapter content.
- Markdown code fences are left untouched.
- The rewrite is idempotent.

