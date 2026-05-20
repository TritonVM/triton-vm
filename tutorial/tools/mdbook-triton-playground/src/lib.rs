//! A custom mdBook preprocessor to make sure that Triton Playground content
//! is formatted correctly.
//!
//! mdBook doesn’t automatically treat content inside the custom
//! `<triton-playground>` tag as code-like text; instead, it applies a rendering
//! pass, inserting `<p>` and other tags.
//! This preprocessor normalizes the markup so Triton playground snippets render
//! consistently (including when embedded in tricky HTML contexts like
//! `<details>`).

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use kuchiki::traits::*;
use mdbook_preprocessor::Preprocessor;
use mdbook_preprocessor::PreprocessorContext;
use mdbook_preprocessor::book::Book;
use mdbook_preprocessor::book::BookItem;
use mdbook_preprocessor::errors::Error as MdBookError;
use pulldown_cmark::Event;
use pulldown_cmark::Parser;

const OPEN_TAG_PREFIX: &str = "<triton-playground";
const CLOSE_TAG: &str = "</triton-playground>";

pub struct TritonPlayPreprocessor;

impl TritonPlayPreprocessor {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for TritonPlayPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor for TritonPlayPreprocessor {
    fn name(&self) -> &str {
        "triton-playground"
    }

    fn run(&self, _: &PreprocessorContext, mut book: Book) -> Result<Book, MdBookError> {
        for item in &mut book.items {
            transform_book_item(item);
        }

        Ok(book)
    }
}

fn transform_book_item(item: &mut BookItem) {
    let &mut BookItem::Chapter(ref mut chapter) = item else {
        return;
    };

    chapter.content = transform_markdown(&chapter.content);
    for sub_item in &mut chapter.sub_items {
        transform_book_item(sub_item);
    }
}

pub fn transform_markdown(input: &str) -> String {
    let mut transformed = String::with_capacity(input.len());
    let mut last_end = 0usize;
    let mut open_widget_start = None;

    for (event, range) in Parser::new(input).into_offset_iter() {
        let html = match event {
            Event::Html(html) | Event::InlineHtml(html) => html,
            _ => continue,
        };
        let html = html.as_ref();

        // Because blank lines end HTML blocks, the content we're interested in
        // might span multiple Events. This is the reason this preprocessor
        // exists in the first place…
        // See also: https://spec.commonmark.org/0.31.2/#html-blocks
        if let Some(open_start) = open_widget_start {
            if let Some(close_end_offset) = close_tag_end_offset(html) {
                let close_end = range.start + close_end_offset;
                transformed.push_str(&input[last_end..open_start]);
                transformed.push_str(&transform_html_event(&input[open_start..close_end]));
                last_end = close_end;
                open_widget_start = None;
            }
            continue;
        }

        let Some(open_start_in_event) = open_tag_start_offset(html) else {
            continue;
        };

        let open_start = range.start + open_start_in_event;
        let Some(close_end_offset) = close_tag_end_offset(&html[open_start_in_event..]) else {
            open_widget_start = Some(open_start);
            continue;
        };

        let close_end = open_start + close_end_offset;
        transformed.push_str(&input[last_end..open_start]);
        transformed.push_str(&transform_html_event(&input[open_start..close_end]));
        last_end = close_end;
    }

    transformed.push_str(&input[last_end..]);
    transformed
}

fn open_tag_start_offset(html: &str) -> Option<usize> {
    html.find(OPEN_TAG_PREFIX)
}

fn close_tag_end_offset(html: &str) -> Option<usize> {
    html.find(CLOSE_TAG).map(|idx| idx + CLOSE_TAG.len())
}

fn transform_html_event(html: &str) -> String {
    if !html.contains(OPEN_TAG_PREFIX) {
        return html.to_owned();
    }

    match rewrite_html_fragment(html) {
        Ok(transformed) => transformed,
        Err(err) => {
            eprintln!("mdbook-triton-playground: leaving HTML fragment unchanged: {err}");
            html.to_owned()
        }
    }
}

fn rewrite_html_fragment(fragment: &str) -> Result<String> {
    let wrapped = format!(r#"<div data-triton-play-root="true">{fragment}</div>"#);
    let document = kuchiki::parse_html().one(wrapped);
    let root = document
        .select_first(r#"div[data-triton-play-root="true"]"#)
        .map_err(|_| anyhow!("missing fragment root"))?
        .as_node()
        .clone();

    for match_node in root
        .select("triton-playground")
        .map_err(|_| anyhow!("failed to select triton-playground nodes"))?
    {
        let node = match_node.as_node().clone();
        wrap_in_code(&node)?;
    }
    let serialized = serialize_children(&root)?;

    Ok(normalize_code_newlines(&serialized))
}

fn wrap_in_code(playground: &kuchiki::NodeRef) -> Result<()> {
    let wrapper = kuchiki::parse_html().one("<code></code>");
    let code = wrapper
        .select_first("code")
        .map_err(|_| anyhow!("missing code wrapper"))?
        .as_node()
        .clone();

    let children: Vec<_> = playground.children().collect();
    for child in children {
        code.append(child);
    }

    playground.append(code);
    Ok(())
}

fn serialize_children(node: &kuchiki::NodeRef) -> Result<String> {
    let mut out = Vec::new();
    for child in node.children() {
        child.serialize(&mut out)?;
    }

    String::from_utf8(out).context("fragment serialization was not valid UTF-8")
}

fn normalize_code_newlines(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0;

    while let Some(code_open_rel) = html[cursor..].find("<code>") {
        let code_open = cursor + code_open_rel;
        let code_text_start = code_open + "<code>".len();
        out.push_str(&html[cursor..code_text_start]);

        let Some(code_close_rel) = html[code_text_start..].find("</code>") else {
            out.push_str(&html[code_text_start..]);
            return out;
        };

        let code_close = code_text_start + code_close_rel;
        for ch in html[code_text_start..code_close].chars() {
            match ch {
                '\n' => out.push_str("&#10;"),
                '\r' => out.push_str("&#13;"),
                _ => out.push(ch),
            }
        }
        out.push_str("</code>");
        cursor = code_close + "</code>".len();
    }

    out.push_str(&html[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn wraps_raw_widget_source_in_code() {
        let input = r#"<triton-playground>
call my_label
halt

my_label:
  // do something
  return

</triton-playground>"#;

        let output = transform_markdown(input);
        assert!(output.contains("<triton-playground>"));
        assert!(output.contains("<code>"));
        assert!(output.contains("</code></triton-playground>"));
        assert!(output.contains("call my_label"));
        assert!(output.contains("return"));
    }

    #[test]
    fn preserves_code_fences_untouched() {
        let input = r#"```html
<triton-playground>
call my_label
halt
</triton-playground>
```"#;

        let output = transform_markdown(input);
        assert_eq!(input, output);
    }

    #[test]
    fn wraps_widget_spanning_multiple_html_events() {
        let input = r#"<triton-playground>
call foo
halt

foo:
  push 1
  return
</triton-playground>"#;

        let output = transform_markdown(input);
        assert!(output.contains("<code>"));
        assert!(output.contains("</code></triton-playground>"));
        assert_eq!(output.matches("</triton-playground>").count(), 1);
    }

    #[test]
    fn preserves_surrounding_html_when_widget_tag_starts_mid_event() {
        let input = r#"<details>
<summary>Hint</summary><triton-playground>
push 1
halt
</triton-playground>
</details>"#;

        let output = transform_markdown(input);
        assert!(output.contains("<summary>Hint</summary>"));
        assert!(output.contains("<code>"));
        assert_eq!(output.matches("<details>").count(), 1);
        assert_eq!(output.matches("</details>").count(), 1);
        assert_eq!(output.matches("<triton-playground>").count(), 1);
        assert_eq!(output.matches("</triton-playground>").count(), 1);
    }

    #[test]
    fn preserves_html_after_widget_close_in_same_event() {
        let input = "<triton-playground>push 1\nhalt\n</triton-playground><br>";
        let output = transform_markdown(input);
        assert!(output.contains("<code>"));
        assert!(output.ends_with("<br>"));
    }

    #[test]
    fn encodes_newlines_inside_code_block() {
        let input = r#"<triton-playground>
push 1

halt
</triton-playground>"#;

        let output = transform_markdown(input);
        assert!(output.contains("<code>&#10;push 1&#10;&#10;halt&#10;</code>"));
        assert!(!output.contains("<code>\npush 1\n\nhalt\n</code>"));
    }
}
